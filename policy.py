import jax
import jax.numpy as np

from model import gen_path, gen_jit, pol0
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

def welfare_path(par, sim, disc, long_run, T):
    # discounting
    ydelt = 1/year_days
    ytvec = np.arange(T)/year_days
    down = np.exp(-disc*ytvec)

    # input factors
    out = sim['out'][:T]
    irate = np.diff(sim['ka'][:T])
    irate = np.concatenate([irate, irate[-1:]])

    # immediate welfare
    util = out - par['ψ']*irate
    welf0 = (ydelt*down*util).sum()

    # total welfare
    if long_run:
        welf1 = (down[-1]*util[-1])/disc
        welf = disc*(welf0 + welf1)
    else:
        Ty = T/year_days
        welf = disc*welf0/(1-np.exp(-disc*Ty))

    return welf

def welfare(pol, par, st0, disc, long_run, T):
    sim = gen_path(par, pol, st0, T)
    return welfare_path(par, sim, disc, long_run, T)

def eval_policy(pol, par, st0, T, disc=0.05, long_run=True):
    pol = {**pol0, **pol}
    sim = gen_jit(par, pol, st0, T)
    return welfare_path(par, sim, disc, long_run, T)

def print_policy(j, val, pol):
    pstr = ', '.join(f'{k}={np.mean(v):.4f}' for k, v in pol.items())
    print(f'[{j:5d}] {val:.4f}: {pstr}')

def optimal_policy(pol, par, st0, T, disc=0.05, long_run=False, hard_policy=hard_default, optim='adam', **kwargs):
    if optim == 'rmsprop':
        opter = rmsprop
    elif optim == 'adam':
        opter = adam

    def objective(lpol):
        pol = trans_args(lpol, spec_policy, hard_policy)
        return welfare(pol, par, st0, disc, long_run, T)

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
