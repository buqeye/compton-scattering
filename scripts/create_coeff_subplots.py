import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from compton import expansion_parameter, expansion_parameter_transfer_cm, order_transition, coefficients
from compton import mass_proton, mass_neutron
from plotly.subplots import make_subplots


# nucleon = 'neutron'
nucleon = 'proton'

df = pd.read_csv('../data/compton_observables.csv', index_col=False)
order_map = {0: 0, 2: 1, 3: -1, 4: 2}
orders = np.array([0, 2, 3, 4])

obs_vals = [
    'crosssection', '1Xp', '1X', '1Zp', '1Z', '2Xp',
    '2X', '2Zp', '2Z', '3', '3Yp', '3Y', 'Y'
]
nrows = len(obs_vals) // 2 + len(obs_vals) % 2
specs = [[{'type': 'surface'}, {'type': 'surface'}] for i in range(nrows)]

fig = make_subplots(
    rows=nrows, cols=2,
    specs=specs,
    subplot_titles=[f'{obs} {nucleon}' for obs in obs_vals],
    horizontal_spacing=0.05,
    vertical_spacing=2e-2,
)

print([f'{obs} {nucleon}' for obs in obs_vals])
scene_idx = 0
for i, obs in enumerate(obs_vals):
    col = i % 2 + 1
    row = i // 2 + 1

    df_obs = df[(df['observable'] == obs) & (df['nucleon'] == nucleon)]
    X = df_obs[['omegalab [MeV]', 'thetalab [deg]']].values
    omega = np.unique(X[:, 0])
    degrees = np.unique(X[:, 1])
    mass = mass_proton if nucleon == 'proton' else mass_neutron

    Lambdab = 650
    # Q = expansion_parameter2(X, Lambdab)
    Q = expansion_parameter_transfer_cm(X, Lambdab, mass)
    ord_vals = np.array([order_transition(order, order_map[order], X[:, 0]) for order in orders]).T
    # ord_vals = orders
    y = df_obs[['y0', 'y2', 'y3', 'y4']].values
    y_grid = y.reshape(len(omega), len(degrees), -1)
    ref = 1.
    if obs == 'crosssection':
        ref = df_obs['y4'].values
    coeffs = coefficients(y, ratio=Q, orders=ord_vals, ref=ref)
    coeffs_grid = coeffs.reshape(len(omega), len(degrees), -1)

    data = []
    names = ['c{}'.format(i) for i in orders]
    for k, name in enumerate(names):
        color = mpl.colors.to_hex('C{}'.format(k))
        z = coeffs_grid[..., k].T
        n_skip = 2
        surf = go.Surface(
            z=z[::n_skip, ::n_skip], x=omega[::n_skip], y=degrees[::n_skip], name=name,
            showscale=False, opacity=1., colorscale=[[0, color], [1, color]],
        )
        fig.add_trace(surf, row=row, col=col)
    if scene_idx == 0:
        scene_i = 'scene'
    else:
        scene_i = f'scene{scene_idx}'
    scene_idx += 1
    fig.update_layout({scene_i: dict(xaxis_title="omega lab [MeV]", yaxis_title="theta lab [deg]", aspectmode='cube')})

fig.update_layout(height=3700, width=1400, title_text="Coefficients")
fig.write_html(f'../scripts/figures_interactive/all_{nucleon}_coeffs.html',
               include_plotlyjs='directory')
