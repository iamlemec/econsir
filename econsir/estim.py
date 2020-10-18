import jax
import jax.numpy as np

import pandas as pd

from .model import zero_state, gen_path
from .tools import (
    load_args, save_args, trans_args, rtrans_args,
    rmsprop, adam, gaussian_err, poisson_err,
)

##
## estimation
##

# parameter transforms
spec_params = {
    'β': 'log',
    'γ': 'log',
    'λ': 'log',
    'δ': 'log',
    'κ': 'log',
    'ϝ': 'log',
    'ψ': 'log',
    'σ': 'log',
    'ρ': 'log',
    'p0': 'logit',
    'pr': 'log',
    'β0': 'logit',
    'βr': 'log',
    'ψ0': 'log',
    'ψr': 'log',
    'zi': 'log',
    'zb1': 'log',
    'zb2': 'log',
    'zb3': 'log',
    'σa': 'log',
    'σo': 'log',
}

# map from policy data to z levels
def calc_zbar(par, dat):
    return (
          dat['pol1']*(1-dat['pol2'])*(1-dat['pol3'])*par['zb1']
        + dat['pol2']*(1-dat['pol3'])*par['zb2']
        + dat['pol3']*par['zb3']
    )

def likelihood(par, dat):
    T, K = dat['T'], dat['K']

    # compute policy
    zb0 = calc_zbar(par, dat)
    pol = {'zb': zb0, 'zc': 0, 'kz': 0, 'vx': 0}

    # tabulate state
    st0 = zero_state(K)
    imp = dat['imp']

    # run simulation
    sim, _ = gen_path(par, pol, st0, imp, T, K)

    # extract simulation
    sim_c = sim['c']
    sim_d = sim['d']
    sim_a = sim['act']
    sim_o = sim['out']

    # extract data
    dat_c = dat['c']
    dat_d = dat['d']
    dat_a = dat['act']
    dat_o = dat['out']

    # get daily rates
    sim_c = np.diff(sim_c, axis=0)
    sim_d = np.diff(sim_d, axis=0)
    dat_c = np.diff(dat_c, axis=0)
    dat_d = np.diff(dat_d, axis=0)

    # actual standard deviations
    wgt0 = dat['wgt'][None, :]
    wgt1 = wgt0/np.mean(wgt0)
    sig_0 = 1/np.sqrt(wgt1)
    sig_a = sig_0*par['σa']
    sig_o = sig_0*par['σo']

    # epi match
    lik_c = poisson_err(dat_c, sim_c, wgt0)
    lik_d = poisson_err(dat_d, sim_d, wgt0)

    # econ match
    lik_a = gaussian_err(dat_a, sim_a, sig_a)
    lik_o = gaussian_err(dat_o, sim_o, sig_o)

    # sum it all up
    lik = 0.2*lik_c + 5*lik_d + lik_a + lik_o

    return lik

def print_params(j, val, par):
    pstr = ', '.join(f'{k}={np.mean(v):.4f}' for k, v in par.items())
    print(f'[{j:5d}] {val:.4f}: {pstr}')

def estimate(par, dat, hard_params={}, optim='adam', save=None, **kwargs):
    if type(par) is str:
        par = load_args(par)

    if optim == 'rmsprop':
        opter = rmsprop
    elif optim == 'adam':
        opter = adam

    def objective(lpar):
        par = trans_args(lpar, spec_params, hard_params)
        return likelihood(par, dat)

    def printer(j, val, lpar):
        par = trans_args(lpar, spec_params, hard_params)
        print_params(j, val, par)

    # gradient and compile
    obj_gradval0 = jax.value_and_grad(objective)
    obj_gradval = jax.jit(obj_gradval0)

    # so we don't waste time on gradients
    par0 = {k: v for k, v in par.items() if k not in hard_params}
    lpar0 = rtrans_args(par0, spec_params)

    # run the optimizer
    lpar1 = opter(obj_gradval, lpar0, disp=printer, **kwargs)
    par1 = trans_args(lpar1, spec_params, hard_params)

    if save is not None:
        save_args(par1, save)

    return par1
