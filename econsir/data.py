import numpy as np
import pandas as pd

##
## data spec
##

data_cols = ['c', 'd', 'act', 'out', 'pol1', 'pol2', 'pol3']

##
## data tools
##

# fill only leading nans with zero
def zfill(ser, val=0):
    ser1 = ser.copy()
    idx1 = ser1.first_valid_index()
    if idx1 is not None and idx1 > ser1.index[0]:
        ser1.loc[:idx1] = val
    return ser1

# winsorize series
def winsorize(s, level=0.05):
    if type(level) not in (list, tuple):
        level = level, 1.0-level
    return s.clip(
        lower=s.quantile(level[0]),
        upper=s.quantile(level[1]),
    )

def get_impulse(df, cutoff=0, axis=0):
    start = (df >= cutoff).astype(np.float)
    imp = start.diff(axis=axis).fillna(0) # just the binary value
    imp.iloc[0] += 1 - imp.sum(axis=axis) # in case it starts before window
    imp = imp*df # get the magnitude
    return imp

def nan_filter(df, idx, col, thresh=0):
    has_col = f'has_{col}'
    qual_col = f'qual_{col}'
    df1 = df[[idx, col]].copy()
    df1[has_col] = df1[col].notna()
    qual = df1.groupby(idx)[has_col].mean().rename(qual_col)
    df1 = df1.join(qual, on=idx)
    return df[df1[qual_col] > thresh]

##
## data loader
##

def load_data(path, cutoff=1e-6, lag=10, smooth=7, winsor=0.05, min_out=0.2, min_pop=50_000, min_date='2020-02-15', max_date=None):
    dtype = {
        'county_fips': 'str', 'pop': 'Int64', 'cases_cum': 'Int64',
        'deaths_cum': 'Int64', 'policy1': 'boolean', 'policy2': 'boolean',
        'policy3': 'boolean'
    }
    data = pd.read_csv(path, dtype=dtype, parse_dates=['date'])
    data = data.dropna(subset=['pop'])

    # select on population
    if min_pop is not None:
        data = data[data['pop'] >= min_pop]

    # select on having any epi data
    data = nan_filter(data, 'county_fips', 'cases_cum')
    data = nan_filter(data, 'county_fips', 'deaths_cum')

    # select on output data quality
    if min_out is not None:
        data = nan_filter(data, 'county_fips', 'out_norm', min_out)

    # per capita rates
    data['cases_pc'] = data['cases_cum']/data['pop']
    data['deaths_pc'] = data['deaths_cum']/data['pop']

    # reshape into panel
    panel = data[[
        'county_fips', 'date', 'cases_pc', 'deaths_pc', 'act_norm', 'out_norm',
        'policy1', 'policy2', 'policy3'
    ]]
    panel = panel.set_index(['county_fips', 'date']).unstack(level='county_fips')

    # fill in cases and deaths
    panel['cases_pc'] = panel['cases_pc'].apply(zfill).fillna(method='ffill')
    panel['deaths_pc'] = panel['deaths_pc'].apply(zfill).fillna(method='ffill')

    # fill in activity and output
    one_fill = lambda s: zfill(s, val=1)
    panel['act_norm'] = panel['act_norm'].apply(one_fill).fillna(method='ffill')
    panel['out_norm'] = panel['out_norm'].apply(one_fill).fillna(method='ffill')

    # assume current policy continues forever
    panel['policy1'] = panel['policy1'].fillna(method='ffill').fillna(0)
    panel['policy2'] = panel['policy2'].fillna(method='ffill').fillna(0)
    panel['policy3'] = panel['policy3'].fillna(method='ffill').fillna(0)

    # smoothing
    if smooth is not None:
        panel = panel.rolling(smooth, min_periods=1).mean()

    # winsorize series
    if winsor is not None:
        wins_func = lambda s: winsorize(s, level=winsor)
        panel['act_norm'] = panel['act_norm'].apply(wins_func, axis=1)
        panel['out_norm'] = panel['out_norm'].apply(wins_func, axis=1)

    # lag epi params
    if lag is not None:
        panel['cases_pc'] = panel['cases_pc'].shift(-lag, freq='1d')
        panel['deaths_pc'] = panel['deaths_pc'].shift(-lag, freq='1d')
        panel = panel.iloc[:-lag]

    # select date ranges
    if min_date is not None:
        panel = panel.loc[min_date:]
    if max_date is not None:
        panel = panel.loc[:max_date]

    # get outbreak start and case impulse
    imp = get_impulse(panel['cases_pc'], cutoff=cutoff)
    panel = panel.join(pd.concat({'impulse': imp}, axis=1))

    # get stats
    wgt = data.groupby('county_fips')['pop'].first().astype(np.float)
    date = panel.index
    fips = panel.columns.levels[1]
    T, = panel.index.shape
    _, K = panel.columns.levshape

    # return all
    return {
        'c': panel['cases_pc'].values,
        'd': panel['deaths_pc'].values,
        'act': panel['act_norm'].values,
        'out': panel['out_norm'].values,
        'pol1': panel['policy1'].values,
        'pol2': panel['policy2'].values,
        'pol3': panel['policy3'].values,
        'imp': panel['impulse'].values,
        'date': date.values,
        'fips': fips.values,
        'wgt': wgt.values,
        'T': T,
        'K': K,
    }
