import jax
import jax.numpy as np
from jax.tree_util import tree_map, tree_leaves
import toml
import numpy as np0
import pandas as pd

##
## toml tools
##

def norm_arg(x):
    if type(x) is jax.interpreters.xla.DeviceArray:
        if x.ndim == 0:
            return float(x)
        else:
            return [float(z) for z in x]
    else:
        return float(x)

def make_arg(x):
    if type(x) is list:
        return np.array(x)
    else:
        return np.float32(x)

def save_args(arg, path):
    arg1 = {k: norm_arg(v) for k, v in arg.items()}
    with open(path, 'w+') as fid:
        toml.dump(arg1, fid)

def load_args(path):
    with open(path) as fid:
        arg = toml.load(fid)
    arg1 = {k: make_arg(v) for k, v in arg.items()}
    return arg1

##
## matrix tools
##

def framify(d, index, columns):
    if not isinstance(index, pd.Index):
        index = pd.DatetimeIndex(index, name='date')
    if not isinstance(columns, pd.Index):
        columns = pd.Index(columns, name='county_fips')
    return pd.concat({
        k: pd.DataFrame(
            v.copy(), index=index, columns=columns
        ) for k, v in d.items() if np.ndim(v) == 2
    }, axis=1)

def one_hot(index, value=1, T=None):
    if T is None:
        T = np0.max(index)
    K = len(index)
    mat = np0.zeros((T, K))
    mat[index, np0.arange(K)] = value
    return mat

##
## argument transformations
##

logit = lambda lv: 1/(1+np.exp(-lv))
rlogit = lambda v: np.log(v/(1-v))
elog = -rlogit(1e-6)

def scaler(z):
    return (
        lambda x: z*np.exp(x),
        lambda x: np.log(x/z),
    )

def trans_args(larg, spec, hard):
    arg = {}
    for k, lv in larg.items():
        s = spec[k]
        if type(s) is tuple:
            v = s[0](lv)
        elif s == 'log':
            v = np.exp(lv)
        elif s == 'log-norm':
            e = np.exp(lv)
            v = e/np.mean(e)
        elif s == 'logit':
            v = logit(lv)
        elif s == 'elogit':
            v = logit(lv-elog)
        elif s == 'ident':
            v = lv
        else:
            raise Exception(f'Unrecognized transform: {s}')
        arg[k] = v
    arg.update(hard)
    return arg

def rtrans_args(arg, spec):
    larg = {}
    for k, v in arg.items():
        s = spec[k]
        if type(s) is tuple:
            lv = s[1](v)
        elif s == 'log' or s == 'log-norm':
            lv = np.log(v)
        elif s == 'logit':
            lv = rlogit(v)
        elif s == 'elogit':
            lv = rlogit(v) + elog
        elif s == 'ident':
            lv = v
        else:
            raise Exception(f'Unrecognized transform: {s}')
        larg[k] = lv
    return larg

##
## diagnostics
##

def summary_naninf(d):
    for k, v in d.items():
        if (na := np.isnan(v)).any():
            print(f'{k} nan: {na.sum()}/{na.size}')
        if (nf := np.isinf(v)).any():
            print(f'{k} inf: {nf.sum()}/{nf.size}')

##
## optimization
##

def clip1(x, c):
    n = np.linalg.norm(x)
    return np.where(n > c, x*(c/n), x)

def clip2(x, c):
    return np.clip(x, -c, c)

# RMSprop gradient maximizer
def rmsprop(gradval, params, eta=0.01, gamma=0.9, eps=1e-8, R=500, per=100, disp=None):
    params = {k: np.array(v) for k, v in params.items()}
    grms = {k: np.zeros_like(v) for k, v in params.items()}
    n = len(params)

    # iterate to max
    for j in range(R+1):
        val, grad = gradval(params)

        vnan = np.isnan(val)
        gnan = tree_map(lambda g: np.isnan(g).any(), grad)

        if vnan or np.any(tree_leaves(gnan)):
            print('Encountered nans!')
            disp(j, val, params)
            return params

        for k in params:
            grms[k] += (1-gamma)*(grad[k]**2-grms[k])
            params[k] += eta*grad[k]/np.sqrt(grms[k]+eps)

        if disp is not None and j % per == 0:
            disp(j, val, params)

    return params

# Adam gradient maximizer
def adam(gradval, params, eta=0.01, beta1=0.9, beta2=0.9, eps=1e-7, c=0.01, R=500, per=100, disp=None, log=False):
    params = {k: np.array(v) for k, v in params.items()}
    gavg = {k: np.zeros_like(v) for k, v in params.items()}
    grms = {k: np.zeros_like(v) for k, v in params.items()}
    n = len(params)

    if log:
        hist = {k: [] for k in params}

    # iterate to max
    for j in range(R+1):
        val, grad = gradval(params)

        # test for nans
        vnan = np.isnan(val) or np.isinf(val)
        gnan = np.array([np.isnan(g).any() or np.isinf(g).any() for g in grad.values()])
        if vnan or gnan.any():
            print('Encountered nan/inf!')
            disp(j, val, params)
            summary_naninf(grad)
            return params

        # clip gradients
        gradc = {k: clip2(grad[k], c) for k in params}

        # early bias correction
        etat = eta*(np.sqrt(1-beta2**(j+1)))/(1-beta1**(j+1))

        for k in params:
            gavg[k] += (1-beta1)*(gradc[k]-gavg[k])
            grms[k] += (1-beta2)*(gradc[k]**2-grms[k])
            params[k] += etat*gavg[k]/(np.sqrt(grms[k])+eps)

            if log:
                hist[k].append(params[k])

        if disp is not None and j % per == 0:
            disp(j, val, params)

    if log:
        hist = {k: np.stack(v) for k, v in hist.items()}
        return hist

    return params

##
## error functions
##

eps = 1e-10

def log(x):
    return np.log(np.maximum(eps, x))

def gaussian_err(dat, sim, sig):
    lik = - log(sig) - 0.5*((dat-sim)/sig)**2
    return np.mean(lik)

def poisson_err(dat, sim, n):
    lik = n * ( dat * log(sim) - sim )
    return np.mean(lik)
