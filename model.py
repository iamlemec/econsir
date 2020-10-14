import jax
import jax.lax as lax
import jax.numpy as np
from jax.scipy.special import ndtr

import pandas as pd

from tools import load_args, log, eps

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

    # params
    β = par['β']
    λ = par['λ']
    γ = par['γ']
    δ = par['δ']
    κ = par['κ']
    ψ = par['ψ']
    σ = par['σ']
    ρ = par['ρ']
    p0 = par['p0']
    pr = par['pr']
    β0 = par['β0']
    βr = par['βr']
    ψ0 = par['ψ0']
    ψr = par['ψr']

    # policy
    zi = par['zi']
    zc = tv['zc']
    zb = tv['zb']
    kz = tv['kz']
    vx = tv['vx']

    # realized policy
    zb1 = killzone(a+i, zc)*zb
    zi1 = np.maximum(zi, zb1)

    # derived z mean
    μ = -0.5*σ**2

    # time varying params
    p = p0 + (1-p0)*smoothstep(pr*t)
    βs = β0 + (1-β0)*smoothstep(βr*t)
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

    # updates
    sp = s + ds
    ap = a + da
    ip = i + di
    rp = r + dr
    dp = d + dd
    cp = c + dc
    kap = ka + dka
    kip = ki + dki
    ep = e + de
    tp = t + dt

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
def gen_path(par, pol, st0, T):
    tv = {
        'zb': pol['zb']*np.ones(T),
        'zc': pol['zc']*np.ones(T),
        'kz': pol['kz']*np.ones(T),
        'vx': pol['vx']*np.ones(T),
    }

    sf = jax.partial(sir, par)
    last, path = lax.scan(sf, st0, tv)

    return {k: v.T for k, v in path.items()}

# null state
def zero_state(par):
    i = par['i0']/1e6
    ϝ = np.maximum(0, (par['β']+par['δ']+par['κ']-par['λ']-par['γ'])/par['λ'])
    p0 = par['p0']
    return {
        's': 1.0 - (1+ϝ)*i,
        'a': ϝ*i,
        'i': i,
        'r': 0.0,
        'd': 0.0,
        'c': p0*i,
        'ka': ϝ*i,
        'ki': i,
        'e': 1.0,
        't': 0.0,
    }

# compiled simulator
gen_jit = jax.jit(gen_path, static_argnums=(3,))

# default policy
pol0 = {
    'zb': 0.0,
    'zc': 0.0,
    'kz': 0.0,
    'vx': 0.0,
}

# simple interface
def simulate_path(par='config/params.toml', pol={}, st0=None, T=365, date='2020-02-15', frame=True):
    if type(par) is str:
        par = load_args(par)
    if st0 is None:
        st0 = zero_state(par)
    pol = {**pol0, **pol}

    sim = gen_jit(par, pol, st0, T)

    if frame:
        index = pd.date_range(date, periods=T, freq='d', name='date')
        return pd.DataFrame(sim, index=index)
    else:
        return sim
