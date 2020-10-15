import pandas as pd

##
## data spec
##

data_cols = ['c', 'd', 'act', 'out', 'pol1', 'pol2', 'pol3']

##
## loader
##

def load_data(path='data/usa.csv', smooth=7, lag=10, date='2020-02-15'):
    dat = pd.read_csv(path, parse_dates=['date']).set_index('date')

    # smoothing
    if smooth is not None:
        dat = dat.rolling(smooth, min_periods=1).mean()

    # lag epi params
    if lag is not None:
        dat['c'] = dat['c'].shift(lag)
        dat['d'] = dat['d'].shift(lag)
        dat = dat.dropna()

    # impose start date
    dat = dat.loc[date:]

    return dat[data_cols]
