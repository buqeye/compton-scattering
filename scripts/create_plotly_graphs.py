import numpy as np
import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from itertools import product

# Make sure these are right
mass_proton = 938.272
alpha_fine = 0.00729735256
hbarc = 197.326
fm2_to_nb = 1e7  # I think


def expansion_parameter(X, breakdown):
    X = np.atleast_2d(X)
    m_pi = 138
    return np.squeeze((X[:, 0] + m_pi) / breakdown)


def thompson_limit(degrees):
    return (alpha_fine * hbarc / mass_proton)**2 * fm2_to_nb * \
        (1 + np.cos(np.pi * degrees / 180)**2) / 2


df = pd.read_csv('../compton_observables.csv', index_col=False)

# obs vals: crosssection, 1Xp, 1X, 1Zp, 1Z, 2Xp, 2X, 2Zp, 2Z, 3, 3Yp, 3Y, Y
obs_vals = [
    'crosssection', '1Xp', '1X', '1Zp', '1Z', '2Xp',
    '2X', '2Zp', '2Z', '3', '3Yp', '3Y', 'Y'
]
# obs_vals = [
#     'crosssection'
# ]
systems = ['neutron', 'proton']

for obs, system in product(obs_vals, systems):
    print(obs, system)
    df_obs = df[(df['observable'] == obs) & (df['system'] == system)]
    X = df_obs[['omegalab [MeV]', 'thetalab [deg]']].values
    omega = np.unique(X[:, 0])
    degrees = np.unique(X[:, 1])

    Lambdab = 600
    Q = expansion_parameter(X, Lambdab)
    orders = np.array([0, 2, 3, 4])
    y = df_obs[['y0', 'y2', 'y3', 'y4']].values
    y_grid = y.reshape(len(omega), len(degrees), -1)
    ref = 1.
    if obs == 'crosssection':
        ref = thompson_limit(X[:, 1].ravel())
    coeffs = gm.coefficients(y, ratio=Q, orders=orders, ref=ref)
    coeffs_grid = coeffs.reshape(len(omega), len(degrees), -1)

    data = []
    names = ['c{}'.format(i) for i in orders]
    for i, name in enumerate(names):
        col = mpl.colors.to_hex('C{}'.format(i))
        z = coeffs_grid[..., i].T
        surf = go.Surface(
            z=coeffs_grid[..., i].T, x=omega, y=degrees, name=name,
            showscale=False, opacity=1., colorscale=[[0, col], [1, col]],
        )
        # This adds a corresponding legend, since surfaces don't have them
        placeholder = go.Scatter3d(
            z=[None], x=[None], y=[None], name=name,
            line=go.scatter3d.Line(color=col),
            surfaceaxis=2, surfacecolor=col,
            showlegend=True, mode='lines',
        )
        data.append(surf)
        data.append(placeholder)

    layout = go.Layout(
        title=f'{obs} {system} coefficients',
        showlegend=True,
        autosize=False,
        width=1100,
        height=700,
        scene=dict(
            xaxis=dict(
                title='Energy',
                range=[omega.min(), np.max(omega)],
            ),
            yaxis=dict(
                title=r'Degrees',
                range=[degrees.min(), degrees.max()],
            ),
            zaxis=dict(
                range=[coeffs_grid.min(), coeffs_grid.max()]
            ),
        ),
    )
    fig = go.FigureWidget(data=data, layout=layout)

    # This is to connect the placeholder legend to the surface
    # To permit toggling surfaces on/off

    # #####################################################################
    # # None of this will work outside of a Jupyter notebook
    # surfaces = {s.name: s for s in fig.data if isinstance(s, go.Surface)}
    # placeholders = {p.name: p for p in fig.data if isinstance(p, go.Scatter3d)}

    # def update_surface_visibility(trace, visible):
    #     surfaces[trace.name].visible = visible

    # [p.on_change(update_surface_visibility, 'visible')
    #  for p in placeholders.values()]
    # #####################################################################

    fig.write_html(f'figures_interactive/coeffs_sys-{system}_obs-{obs}.html',
                   include_plotlyjs='directory')
