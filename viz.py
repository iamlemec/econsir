import toml
import numpy as np
import pandas as pd
import altair as alt

##
## colors
##

neon_blue = '#1e88e5'
neon_red = '#ff0d57'

fg_color = '#bbbbbb'
bg_color = '#212121'

##
## config
##

config = {
    'background': 'none',

    'view': {
        'strokeWidth': 0,
    },

    'axis': {
        'grid': False,
        'domainColor': fg_color,
        'tickColor': fg_color,
        'titleFontSize': 14,
        'labelFontSize': 14,
        'labelColor': fg_color,
    },

    'legend': {
        'titleFontSize': 14,
        'labelFontSize': 14,
    },

    'title': {
        'color': fg_color,
    }
}

alt.themes.register('clean', lambda: {'config': config})
alt.themes.enable('clean')
alt.renderers.set_embed_options(actions=False)

##
## plotting
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

def outcome_summary(df, c_lim=0, d_lim=0, ao_lim=110, ao_base=True):
    c = 1e6*df['c'].diff()
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
            strokeDash=[5, 2], strokeWidth=1, color=fg_color
        ).encode(
            x='date', y='base'
        )
        ch_a += ch_ab

        o_base = pd.DataFrame({'base': base}, index=o.index).reset_index()
        ch_ob = alt.Chart(o_base).mark_line(
            strokeDash=[5, 2], strokeWidth=1, color=fg_color
        ).encode(
            x='date', y='base'
        )
        ch_o += ch_ob

    return ch_c, ch_d, ch_a, ch_o
