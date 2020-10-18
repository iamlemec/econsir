import toml
import numpy as np
import pandas as pd
import altair as alt

##
## colors
##

neon_blue = '#1e88e5'
neon_red = '#ff0d57'

##
## config
##

config = {
    'background': 'transparent',

    'view': {
        'strokeWidth': 0,
    },

    'axis': {
        'grid': False,
        'titleFontSize': 14,
        'labelFontSize': 14,
    },

    'legend': {
        'titleFontSize': 14,
        'labelFontSize': 14,
    },
}

alt.themes.register('clean', lambda: {'config': config})
alt.themes.enable('clean')
alt.renderers.set_embed_options(actions=False)

##
## single path
##

def plot_path(s, title=None, y_max=None):
    if y_max is None:
        y_args = {}
    else:
        y_args = {'scale': alt.Scale(domain=(0, y_max), clamp=True)}

    df = pd.DataFrame({'vals': s}).reset_index()

    ch = alt.Chart(df, width=265, height=150, title=title)
    ch = ch.mark_line(
        color=neon_blue
    ).encode(
        x=alt.X('date', title=None),
        y=alt.Y('vals', title=None, **y_args),
    )

    return ch

def single_summary(df, c_lim=0, d_lim=0, ao_lim=110, ao_base=True, tcase=False, color='black', hspacing=10, vspacing=20):
    c_var = 'ki' if tcase else 'c'

    c = 1e6*df[c_var].diff()
    d = 1e6*df['d'].diff()
    a = 1e2*df['act']
    o = 1e2*df['out']

    c_max = np.maximum(c_lim, c.max())
    d_max = np.maximum(d_lim, d.max())

    ch_c = plot_path(c, title='Daily cases per million', y_max=c_max)
    ch_d = plot_path(d, title='Daily deaths per million', y_max=d_max)
    ch_a = plot_path(a, title='Economic activity (%)', y_max=ao_lim)
    ch_o = plot_path(o, title='Economic output (%)', y_max=ao_lim)

    if ao_base:
        base = 99.4*np.ones(len(df))

        a_base = pd.DataFrame({'base': base}, index=a.index).reset_index()
        ch_ab = alt.Chart(a_base).mark_line(
            strokeDash=[5, 2], strokeWidth=1, color=color
        ).encode(
            x='date', y='base'
        )
        ch_a += ch_ab

        o_base = pd.DataFrame({'base': base}, index=o.index).reset_index()
        ch_ob = alt.Chart(o_base).mark_line(
            strokeDash=[5, 2], strokeWidth=1, color=color
        ).encode(
            x='date', y='base'
        )
        ch_o += ch_ob

    ch = alt.vconcat(
        alt.hconcat(ch_c, ch_d, spacing=hspacing),
        alt.hconcat(ch_a, ch_o, spacing=hspacing),
        spacing=vspacing,
    )

    ch = ch.configure_title(color=color)
    ch = ch.configure_axis(domainColor=color, tickColor=color, labelColor=color)
    ch = ch.configure_axisY(minExtent=30, labelFlush=True)

    return ch

##
## path distribution
##

def gen_quantiles(s, quants=9, qmin=0.05, qmax=0.4):
    if quants is None:
        quants = 25

    if type(quants) is int:
        quants = np.linspace(qmin, qmax, quants)

    qpairs = [(0.5 - q, 0.5 + q) for q in quants]
    qlist = sum([list(p) for p in qpairs], [])

    qvals = s.quantile(qlist, axis=1).T

    return qpairs, qvals

def path_dist(s, wgt=None, y_max=None, title=None, quants=5):
    if y_max is None:
        y_args = {}
    else:
        y_args = {'scale': alt.Scale(domain=(0, y_max), clamp=True)}

    qps, dfs = gen_quantiles(s, quants=quants)

    chs = None
    for qlo, qhi in qps:
        df = dfs[[qlo, qhi]].rename(columns={qlo: 'lower', qhi: 'upper'}).reset_index()
        ch = alt.Chart(df, width=250, height=150, title=title)
        ch = ch.mark_area(color=neon_blue, opacity=0.3)
        ch = ch.encode(
            x=alt.X('date', title=None),
            y=alt.Y('lower', title=None, **y_args),
            y2='upper',
        )
        if chs is None:
            chs = ch
        else:
            chs += ch

    if wgt is not None:
        mean = s.mul(wgt, axis=1).sum(axis=1, min_count=1)/wgt.sum()
    else:
        mean = s.mean(axis=1)

    df = pd.DataFrame({'mean': mean}).reset_index()
    ch = alt.Chart(df).mark_line(color=neon_red).encode(
        x=alt.X('date', title=None),
        y=alt.Y('mean', title=None, **y_args),
    )
    chs += ch

    return chs

def outcome_summary(df, c_lim=0, d_lim=0, ao_lim=110, ao_base=True, tcase=False, split=False, color='black', **kwargs):
    c_var = 'ki' if tcase else 'c'

    c = 1e6*df[c_var].diff()
    d = 1e6*df['d'].diff()
    a = 1e2*df['act']
    o = 1e2*df['out']

    c_max = np.maximum(c_lim, c.quantile(0.9, axis=1).max())
    d_max = np.maximum(d_lim, d.quantile(0.9, axis=1).max())

    ch_c = path_dist(c, title='Daily cases per million', y_max=c_max, **kwargs)
    ch_d = path_dist(d, title='Daily deaths per million', y_max=d_max, **kwargs)
    ch_a = path_dist(a, title='Economic activity (%)', y_max=ao_lim, **kwargs)
    ch_o = path_dist(o, title='Economic output (%)', y_max=ao_lim, **kwargs)

    if ao_base:
        base = 99.4*np.ones(len(df))

        a_base = pd.DataFrame({'base': base}, index=a.index).reset_index()
        ch_ab = alt.Chart(a_base).mark_line(
            strokeDash=[5, 2], strokeWidth=1, color=color
        ).encode(
            x='date', y='base'
        )
        ch_a += ch_ab

        o_base = pd.DataFrame({'base': base}, index=o.index).reset_index()
        ch_ob = alt.Chart(o_base).mark_line(
            strokeDash=[5, 2], strokeWidth=1, color=color
        ).encode(
            x='date', y='base'
        )
        ch_o += ch_ob

    if split:
        return ch_c, ch_d, ch_a, ch_o
    else:
        ch = alt.vconcat(
            alt.hconcat(ch_c, ch_d, spacing=20),
            alt.hconcat(ch_a, ch_o, spacing=20),
            spacing=20,
        )
        ch = ch.configure_axisY(minExtent=40, labelFlush=True)
        ch = ch.configure_axis(domainColor=color, tickColor=color)
        return ch
