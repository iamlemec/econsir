import jax
import jax.lax as lax
import jax.numpy as np
from jax.scipy.special import ndtr

import pandas as pd

from .tools import load_args, framify, log, eps

# β - transmission rate
# λ - infection rate
# γ - asymp recovery rate
# δ - infect recovery rate
# κ - death rate
# ψ - cost of infection
# σ - z distribution shape
# ρ - activity inattention

# zi - quarantine of infected
# zb - lockdown severity
# zc - lockdown cutoff
# kz - killzone threshhold

##
## simulation
##

# log(z) ~ N(μ, σ**2)
def actout(cut, μ, σ):
    avg = np.exp(μ+0.5*σ**2)
    lcut = log(cut)
    act = 1 - ndtr((lcut-μ)/σ)
    out = avg*ndtr((μ+σ**2-lcut)/σ)
    return act, out

# continuous bounded ramp-up
def smoothstep(x):
    return np.where(x > 0, np.where(x < 1, 3*x**2 - 2*x**3, 1), 0)

# rapid die-off around zero
def killzone(x, k, eps=1e-6):
    return smoothstep(1e6*x/(2*(k+eps)))

# core SIR model
def sir(par, st, tv):
    # params
    β = par['β']
    λ = par['λ']
    γ = par['γ']
    δ = par['δ']
    κ = par['κ']
    ψ = par['ψ']
    σ = par['σ']
    ρ = par['ρ']
    ϝ = par['ϝ']
    zi = par['zi']
    p0 = par['p0']
    pr = par['pr']
    β0 = par['β0']
    βr = par['βr']
    ψ0 = par['ψ0']
    ψr = par['ψr']

    # current state
    s = st['s']
    a = st['a']
    i = st['i']
    r = st['r']
    d = st['d']
    c = st['c']
    ka = st['ka']
    ki = st['ki']
    e = st['e']
    t = st['t']

    # impulse
    im = tv['im']

    # policy
    zb = tv['zb']
    zc = tv['zc']
    kz = tv['kz']
    vx = tv['vx']

    # realized policy
    zb1 = killzone(a+i, zc)*zb
    zi1 = np.maximum(zi, zb1)

    # derived z mean
    μ = -0.5*σ**2

    # time varying params
    p = p0 + (1-p0)*smoothstep(pr*t)
    βs = 1 - β0*smoothstep(βr*t)
    ψs = ψ0 + (1-ψ0)*smoothstep(ψr*t)

    # effective params
    β1 = βs*β
    ψ1 = ψs*ψ

    # exogenous excursions
    iact, iout = actout(zi1, μ, σ)
    ract, rout = actout(zb1, μ, σ)

    # endogenous excursions
    prob = β1*(e*a+iact*i)
    zcut = prob*ψ1
    pcut = np.maximum(zb1, zcut)
    sact, sout = actout(pcut, μ, σ)

    # total act/out rates
    act = sact*(s+a) + iact*i + ract*r
    out = sout*(s+a) + iout*i + rout*r

    # flows
    f_sa = prob*sact*s
    f_sr = vx*s
    f_ai = λ*a
    f_ar = γ*a
    f_ir = δ*i
    f_id = κ*i

    # differentials
    ds = - f_sa - f_sr
    da = f_sa - f_ai - f_ar
    di = f_ai - f_ir - f_id
    dr = f_ar + f_ir + f_sr
    dd = f_id
    dc = p*f_ai
    dka = f_sa
    dki = f_ai
    de = ρ*(sact-e)
    dt = 1

    # discrete jumps
    im1 = im/p
    js = -(1+ϝ)*im1
    ja = ϝ*im1
    ji = im1
    jr = 0
    jd = 0
    jc = im
    jka = ϝ*im1
    jki = im1
    je = 0
    jt = 0

    # updates
    sp = s + ds + js
    ap = a + da + ja
    ip = i + di + ji
    rp = r + dr + jr
    dp = d + dd + jd
    cp = c + dc + jc
    kap = ka + dka + jka
    kip = ki + dki + jki
    ep = e + de + je
    tp = t + dt + jt

    # da kill zone
    zz = np.where(kz > 0, killzone(ap+ip, kz), 1)
    ap = zz*ap
    ip = zz*ip

    # new state
    stp = {
        's': sp,
        'a': ap,
        'i': ip,
        'r': rp,
        'd': dp,
        'c': cp,
        'ka': kap,
        'ki': kip,
        'e': ep,
        't': tp,
    }

    # save stats
    xt = stp.copy()
    xt['act'] = act
    xt['out'] = out
    xt['zb1'] = zb1
    xt['βs'] = βs
    xt['ψs'] = ψs
    xt['p'] = p

    return stp, xt

# iterate SIR for path
def gen_path(par, pol, st0, imp, T, K):
    tv = {
        'zb': pol['zb']*np.ones((T, 1)),
        'zc': pol['zc']*np.ones((T, 1)),
        'kz': pol['kz']*np.ones((T, 1)),
        'vx': pol['vx']*np.ones((T, 1)),
        'im': imp,
    }

    sf = jax.partial(sir, par)
    last, path = lax.scan(sf, st0, tv)

    return path, last

# null state
def zero_state(K):
    return {
        's': np.ones(K),
        'a': np.zeros(K),
        'i': np.zeros(K),
        'r': np.zeros(K),
        'd': np.zeros(K),
        'c': np.zeros(K),
        'ka': np.zeros(K),
        'ki': np.zeros(K),
        'e': np.ones(K),
        't': 0,
    }

# compiled simulator
gen_jit = jax.jit(gen_path, static_argnums=(4, 5))

# default policy
pol0 = {
    'zb': 0.0,
    'zc': 0.0,
    'kz': 0.0,
    'vx': 0.0,
}

# simple interface
def simulate_path(par, pol={}, st0=None, imp=None, K=10, T=365, locs=None, date='2020-02-01', frame=True):
    if type(par) is str:
        par = load_args(par)
    if st0 is None:
        st0 = zero_state(K)
    if imp is None:
        imp = np.zeros((T, K))

    pol = {**pol0, **pol}
    sim, last = gen_jit(par, pol, st0, imp, T, K)

    if frame:
        index = pd.date_range(date, periods=T, freq='d', name='date')
        columns = np.arange(K) if locs is None else locs
        return framify(sim, index, columns), last
    else:
        return sim, last
