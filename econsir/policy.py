import jax
import jax.numpy as np

from .model import gen_path, gen_jit, pol0
from .tools import trans_args, rtrans_args, adam, rmsprop

##
## constants
##

year_days = 365.25

##
## optimal policy
##

spec_policy = {
    'zb': 'log',
    'zc': 'log',
    'kz': 'log',
    'vx': 'log',
}

hard_default = {
    'kz': 0,
    'vx': 0,
}

def welfare_path(par, sim, wgt, disc, long_run, T, K):
    # params
    ψ = np.atleast_2d(par['ψ']) # optional county dependence
    wgt1 = wgt/np.sum(wgt) # to distribution

    # discounting
    ydelt = 1/year_days
    ytvec = np.arange(T)/year_days
    down = np.exp(-disc*ytvec)

    # input factors
    out = sim['out'][:T, :]
    irate = np.diff(sim['ka'][:T, :], axis=0)
    irate = np.concatenate([irate, irate[-1:, :]], axis=0)

    # immediate welfare
    util = out - ψ*irate
    eutil = np.sum(util*wgt1[None, :], axis=1)
    welf0 = (ydelt*down*eutil).sum()

    # total welfare
    if long_run:
        welf1 = (down[-1]*eutil[-1])/disc
        welf = disc*(welf0 + welf1)
    else:
        Ty = T/year_days
        welf = disc*welf0/(1-np.exp(-disc*Ty))

    return welf

def welfare(pol, par, st0, imp, wgt, disc, long_run, T, K):
    sim, _ = gen_path(par, pol, st0, imp, T, K)
    welf = welfare_path(par, sim, wgt, disc, long_run, T, K)
    return welf

def eval_policy(pol, par, st0=None, imp=None, wgt=None, T=None, K=None, disc=0.05, long_run=True):
    if st0 is None:
        st0 = zero_state(K)
    if imp is None:
        imp = np.zeros((T, K))
    if wgt is None:
        wgt = np.ones(K)

    pol = {**pol0, **pol}
    sim, _ = gen_jit(par, pol, st0, imp, T, K)
    welf = welfare_path(par, sim, wgt, disc, long_run, T, K)

    return welf

def print_policy(j, val, pol):
    pstr = ', '.join(f'{k}={np.mean(v):.4f}' for k, v in pol.items())
    print(f'[{j:5d}] {val:.4f}: {pstr}')

def optimal_policy(pol, par, st0=None, imp=None, wgt=None, T=None, K=None, disc=0.05, long_run=False, hard_policy=hard_default, optim='adam', **kwargs):
    if st0 is None:
        st0 = zero_state(K)
    if imp is None:
        imp = np.zeros((T, K))
    if wgt is None:
        wgt = np.ones(K)

    if optim == 'rmsprop':
        opter = rmsprop
    elif optim == 'adam':
        opter = adam

    def objective(lpol):
        pol = trans_args(lpol, spec_policy, hard_policy)
        return welfare(pol, par, st0, imp, wgt, disc, long_run, T, K)

    def printer(j, val, lpol):
        pol = trans_args(lpol, spec_policy, hard_policy)
        print_policy(j, val, pol)

    # gradient and compile
    obj_gradval0 = jax.value_and_grad(objective)
    obj_gradval = jax.jit(obj_gradval0)

    # so we don't waste time on gradients
    pol0 = {k: v for k, v in pol.items() if k not in hard_policy}
    lpol0 = rtrans_args(pol0, spec_policy)

    # run the optimizer
    lpol1 = opter(obj_gradval, lpol0, disp=printer, **kwargs)
    pol1 = trans_args(lpol1, spec_policy, hard_policy)

    return pol1
