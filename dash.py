import jax
import econsir as es

import toml
import time
import json
import argparse

import altair as alt
import pandas as pd
import numpy as np

import tornado.ioloop
import tornado.web

from jax.scipy.special import ndtri

# command line arguments
ap = argparse.ArgumentParser(description='Econ-SIR dashboard server')
ap.add_argument('--port', type=int, default=80, help='port to serve on')
ap.add_argument('--params', type=str, default='params/estim.toml', help='parameter set to use')
ap.add_argument('--data', type=str, default='data/panel_usa.csv', help='data file to use')
args = ap.parse_args()

# static params
kill_zone = 400
vax_time = 365
vax_rate = 2/330

# time path
T = 686
t_vec = np.arange(T)[:, None]

# generate dates
base_date = '2020-02-15'
dates = pd.date_range(base_date, periods=T, freq='d', name='date')

# load data
data = es.load_data(args.data, min_date=base_date)
T0, K = data['T'], data['K']
fips = data['fips']
extrap = lambda x, v: np.concatenate([x, v*np.ones((T-T0, K))], axis=0)

# estimated params
params = es.load_args(args.params)

# state path
state = es.zero_state(K)
impul = extrap(data['imp'], 0)

# policy series - extrapolated
pdata = {
    'pol1': extrap(data['pol1'], data['pol1'][[-1], :]),
    'pol2': extrap(data['pol2'], data['pol2'][[-1], :]),
    'pol3': extrap(data['pol3'], data['pol3'][[-1], :]),
}
zbase = es.calc_zbar(params, pdata)

##
## simulate!
##

def sim_policy(act, cut, lag, tcase, kzone, vax):
    σ = params['σ']
    γ = params['γ']
    λ = params['λ']
    δ = params['δ']

    # map from activity to zbar
    act1 = np.maximum(0.001, act/100)
    lz = σ*ndtri(1-act1) - σ**2/2
    zbar = np.exp(lz)

    # map from cases per million per day to zcut
    s_frac = λ/(γ+λ)
    i_time = 1/δ
    zcut = 0.5*cut*i_time/s_frac

    # killzone level
    kval = kill_zone if kzone else 0

    # vaccine rate
    vrat = vax_rate if vax else 0

    # implement time lag
    zbar_vec = np.where(t_vec > lag, zbar, zbase)
    zcut_vec = np.where(t_vec > lag, zcut, 0)
    kzon_vec = np.where(t_vec > lag, kval, 0)
    vax_vec = np.where(t_vec > vax_time, vrat, 0)

    # run simulatrix
    policy = {
        'zb': zbar_vec,
        'zc': zcut_vec,
        'kz': kzon_vec,
        'vx': vax_vec,
    }
    simul, _ = es.gen_jit(params, policy, state, impul, T, K)

    # maybe swap in true cases
    if tcase:
        simul['c'] = simul['ki']

    return es.framify(
        {k: simul[k].copy() for k in ['c', 'd', 'act', 'out']}, dates, fips
    )

##
## server
##

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class PolicyHandler(tornado.web.RequestHandler):
    def get(self):
        act = float(self.get_query_argument('act'))
        cut = float(self.get_query_argument('cut'))
        lag = int(self.get_query_argument('lag'))
        tcase = self.get_query_argument('tcase', 'true') == 'true'
        kzone = self.get_query_argument('kzone', 'true') == 'true'
        vax = self.get_query_argument('vax', 'true') == 'true'
        print(act, cut, lag, tcase, kzone, vax)

        # simulate policy
        t0 = time.time()
        sim_df = sim_policy(act, cut, lag, tcase, kzone, vax)
        t1 = time.time()
        print(f'sim_policy: {t1-t0}')

        # render output
        ch = es.outcome_summary(
            sim_df, c_lim=400, d_lim=8,
            hspacing=20, vspacing=40, color='#bbbbbb',
        )

        # send off vega json
        js = ch.to_json()
        self.write(js)

# start up tornado
handlers = [
    (r'/', MainHandler),
    (r'/policy', PolicyHandler),
]
app = tornado.web.Application(
    handlers,
    static_path='dash',
    template_path='dash',
    debug=True,
)
app.listen(args.port)

print('launching')
tornado.ioloop.IOLoop.current().start()
