import jax
import jax.numpy as np

from model import gen_path
from tools import trans_args, rtrans_args, adam, rmsprop

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

def welfare_path(par, sim, dat, disc, long_run, T, H):
    # discounting
    ydelt = 1/year_days
    ytvec = np.arange(T)/year_days
    down = np.exp(-disc*ytvec)

    # input factors
    out = sim['out']
    irate = np.diff(sim['ke'])
    irate = np.concatenate([irate, irate[-1:]])

    # immediate welfare
    util = out - par['ψ']*irate
    welf0 = (ydelt*down*eutil)[:H].sum()

    # total welfare
    if long_run:
        welf1 = (down[-1]*eutil[-1])/disc
        welf = disc*(welf0 + welf1)
    else:
        Hy = H/year_days
        welf = disc*welf0/(1-np.exp(-disc*Hy))

    return welf

def welfare(pol, par, dat, disc, long_run, K, T, H):
    sim = gen_path(par, pol, dat, K, T)
    return welfare_path(par, sim, dat, disc, long_run, T, H)

def print_policy(j, val, pol):
    pstr = ', '.join(f'{k}={np.mean(v):.4f}' for k, v in pol.items())
    print(f'[{j:5d}] {val:.4f}: {pstr}')

def optimal_policy(pol0, par, dat, K, T, disc=0.05, long_run=False, hard_policy=hard_default, optim='adam', **kwargs):
    if optim == 'rmsprop':
        opter = rmsprop
    elif optim == 'adam':
        opter = adam

    def objective(lpol):
        pol = trans_args(lpol, spec_policy, hard_policy)
        return welfare(pol, par, dat, disc, long_run, K, T, T)

    def printer(j, val, lpol):
        pol = trans_args(lpol, spec_policy, hard_policy)
        print_policy(j, val, pol)

    # gradient and compile
    obj_gradval0 = jax.value_and_grad(objective)
    obj_gradval = jax.jit(obj_gradval0)

    # so we don't waste time on gradients
    pol0 = {k: v for k, v in pol0.items() if k not in hard_policy}
    lpol0 = rtrans_args(pol0, spec_policy)

    # run the optimizer
    lpol1 = opter(obj_gradval, lpol0, disp=printer, **kwargs)
    pol1 = trans_args(lpol1, spec_policy, hard_policy)

    return pol1
