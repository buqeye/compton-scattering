# import numpy as np
# # import gsum as gm
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly as py
# import plotly.graph_objs as go
# import plotly.express as px
# from itertools import product
#
# from compton import mass_proton, alpha_fine, hbarc, fm2_to_nb
# from compton import dsg_proton_low_energy, sigma3_low_energy, expansion_parameter_transfer_cm, expansion_parameter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly as py
# import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import scipy as sp
from scipy.special import expit

from compton import expansion_parameter, expansion_parameter_transfer_cm, order_transition, coefficients
from compton import mass_proton, mass_neutron

from os import path

from compton import create_observable_set


df = pd.read_csv('../data/compton_observables.csv', index_col=False)

obs_file = path.abspath('../data/polarisabilities-coefficient-table-for-all-observables_20191111_jam.csv')
df_ratio = pd.read_csv(obs_file, dtype={'observable': str})
compton_obs = create_observable_set(df=df_ratio, cov_exp=0.)

# obs vals: crosssection, 1Xp, 1X, 1Zp, 1Z, 2Xp, 2X, 2Zp, 2Z, 3, 3Yp, 3Y, Y
obs_vals = [
    'crosssection', '1Xp', '1X', '1Zp', '1Z', '2Xp',
    '2X', '2Zp', '2Z', '3', '3Yp', '3Y', 'Y'
]
# obs_vals = [
#     'crosssection'
# ]
systems = ['neutron', 'proton']
order_map = {0: 0, 2: 1, 3: -1, 4: 2}
orders = np.array([0, 2, 3, 4])

for obs, system in product(obs_vals, systems):
    print(obs, system)
    # df_obs = df[(df['observable'] == obs) & (df['system'] == system)]
    # X = df_obs[['omegalab [MeV]', 'thetalab [deg]']].values
    # omega = np.unique(X[:, 0])
    # degrees = np.unique(X[:, 1])
    # cos0_lab = np.cos(np.deg2rad(X[:, 1]))
    #
    # Lambdab = 600
    # Q = expansion_parameter(X, Lambdab)
    # orders = np.array([0, 2, 3, 4])
    # y = df_obs[['y0', 'y2', 'y3', 'y4']].values
    # y_grid = y.reshape(len(omega), len(degrees), -1)
    # ref = 1.
    # if obs == 'crosssection' and system == 'proton':
    #     # ref = thompson_limit(X[:, 1].ravel())
    #     ref = dsg_proton_low_energy(cos0_lab.ravel())
    # coeffs = coefficients(y, ratio=Q, orders=orders, ref=ref)
    # coeffs_grid = coeffs.reshape(len(omega), len(degrees), -1)

    df_obs = df[(df['observable'] == obs) & (df['nucleon'] == system)]
    # df_obs = df[(df['is_numerator'] is True) & (df['order'] == 4)]

    # df_obs = df_rescaled[(df_rescaled['observable'] == obs) & (df_rescaled['nucleon'] == system)]
    X = df_obs[['omegalab [MeV]', 'thetalab [deg]']].values
    omega = np.unique(X[:, 0])
    degrees = np.unique(X[:, 1])

    mass = mass_proton if system == 'proton' else mass_neutron

    Lambdab = 650
    # Q = expansion_parameter2(X, Lambdab)
    Q = expansion_parameter_transfer_cm(X, Lambdab, mass)
    ord_vals = np.array([order_transition(order, order_map[order], X[:, 0]) for order in orders]).T
    # ord_vals = orders
    y = df_obs[['y0', 'y2', 'y3', 'y4']].values

    # Replace with different LEC values
    from compton import proton_pol_vec_mean, neutron_pol_vec_mean
    p0 = proton_pol_vec_mean if system == 'proton' else neutron_pol_vec_mean
    y[:, 2] = compton_obs[obs, system, 3, 'nonlinear'].prediction_ratio(p0)
    y[:, 3] = compton_obs[obs, system, 4, 'nonlinear'].prediction_ratio(p0)

    y_grid = y.reshape(len(omega), len(degrees), -1)
    ref = 1.
    if obs == 'crosssection':
        ref = df_obs['y4'].values
    coeffs = coefficients(y, ratio=Q, orders=ord_vals, ref=ref)
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
