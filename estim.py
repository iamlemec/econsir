import jax
import jax.numpy as np

from tools import trans_args, rtrans_args, rmsprop, adam, gaussian_err

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
    'ψ': 'log',
    'σ': 'log',
    'ρ': 'log',
    'p0': 'logit',
    'pr': 'log',
    'β0': 'logit',
    'βr': 'log',
    'ψ0': 'logit',
    'ψr': 'log',
    'i0': 'log',
    'zi': 'log',
    'zb1': 'log',
    'zb2': 'log',
    'σc': 'log',
    'σd': 'log',
    'σa': 'log',
    'σo': 'log',
}

def calc_policy(pol1, pol2, zb1, zb2):
    return pol1*(1-pol2)*zb1 + pol2*zb2

def likelihood(params, data):
    st0 = zero_state(par)
    zb0 = calc_policy(data['pol1'], data['pol2'], params['zb1'], params['zb2'])

    # run simulation
    par = params.copy()
    pol = {'zb': zb0, 'zc': 0, 'kz': 0, 'vx': 0}
    sim = gen_path(par, pol, st0, data['T'])

    # extract simulation
    sim_c = 1e6*sim['c']
    sim_d = 1e6*sim['d']
    sim_a = 1e2*sim['act']
    sim_o = 1e2*sim['out']

    # extract data
    dat_c = 1e6*data['c']
    dat_d = 1e6*data['d']
    dat_a = 1e2*data['act']
    dat_o = 1e2*data['out']

    # get daily rates
    sim_c = np.diff(sim_c)
    sim_d = np.diff(sim_d)
    dat_c = np.diff(dat_c)
    dat_d = np.diff(dat_d)

    # actual standard deviations
    sig_c = par['σc']
    sig_d = par['σd']
    sig_a = par['σa']
    sig_o = par['σo']

    # epi match
    lik_c = gaussian_err(dat_c, sim_c, sig_c)
    lik_d = gaussian_err(dat_d, sim_d, sig_d)

    # econ match
    lik_a = gaussian_err(dat_a, sim_a, sig_a)
    lik_o = gaussian_err(dat_o, sim_o, sig_o)

    # sum it all up
    lik = lik_c + lik_d + lik_a + lik_o

    return lik

def print_params(j, val, par):
    pstr = ', '.join(f'{k}={np.mean(v):.4f}' for k, v in par.items())
    print(f'[{j:5d}] {val:.4f}: {pstr}')

def estimate(data, par0, hard_params={}, optim='adam', **kwargs):
    if optim == 'rmsprop':
        opter = rmsprop
    elif optim == 'adam':
        opter = adam

    def objective(lpar):
        par = trans_args(lpar, spec_params, hard_params)
        return like_gradval(par, data)

    def printer(j, val, lpar):
        par = trans_args(lpar, spec_params, hard_params)
        print_params(j, val, par)

    # gradient and compile
    obj_gradval0 = jax.value_and_grad(objective)
    obj_gradval = jax.jit(obj_gradval0)

    # so we don't waste time on gradients
    par0 = {k: v for k, v in par0.items() if k not in hard_params}
    lpar0 = rtrans_args(par0, spec_params)

    # run the optimizer
    lpar1 = opter(objective, lpar0, disp=printer, **kwargs)
    par1 = trans_args(lpar1, spec_params, hard_params)

    return par1
