import matplotlib.pyplot as plt
import numpy as np


def plot_subsets(util_dict, nucleon, omega, degrees, observables=None, subsets=None):
    nucleons_all, observables_all, subsets_all = tuple(zip(*util_dict.keys()))
    nucleons = np.unique(nucleons_all)
    if observables is None:
        observables = np.unique(observables_all)
    if subsets is None:
        subsets = np.unique(subsets_all)

    n_omega = len(omega)
    n_degrees = len(degrees)

    if nucleon not in nucleons:
        raise ValueError('nucleon must be in dict keys')
    fig, axes = plt.subplots(len(observables), len(subsets), figsize=(7, 14), sharex=True, sharey=True)
    for i, obs_i in enumerate(observables):
        # Same colors within rows
        # max_util = np.max([util_dict_trans_trunc_sets[obs_i, subset_name] for subset_name in subsets.keys()])
        # min_util = np.min([util_dict_trans_trunc_sets[obs_i, subset_name] for subset_name in subsets.keys()])

        # Same colors for all plots
        #     max_util = np.max([u for u in util_dict_trans_trunc_sets.values()])
        #     min_util = np.min([u for u in util_dict_trans_trunc_sets.values()])
        #     if i == 0:
        #         print(max_util)
        #         print(np.exp(max_util))
        #     min_util = np.exp(min_util)
        #     max_util = np.exp(max_util)

        # Each plot has own colors
        max_util = None
        min_util = None

        for j, subset_name in enumerate(subsets):
            # Same colors within columns
            #             max_util = np.max([util_dict_trans_trunc_sets[nucleon, o, subset_name] for o in observables_unique])
            #             min_util = np.min([util_dict_trans_trunc_sets[nucleon, o, subset_name] for o in observables_unique])
            #             min_util = np.exp(min_util)
            #             max_util = np.exp(max_util)

            ax = axes[i, j]
            util_i = util_dict[nucleon, obs_i, subset_name]
            #             util_i = np.exp(util_i)

            levels = None
            #             levels = np.linspace(np.floor(min_util), max_util, 11)
            if i == 0:
                pass
            #                 print(subset_name, min_util, max_util, levels)
            #         print(levels)

            ax.contourf(
                omega, degrees, util_i.reshape(n_omega, n_degrees).T,
                vmin=min_util, vmax=max_util,
                #                 locator=ticker.LogLocator(),
                #             levels=np.log(np.arange(1, max_util+0.25, 0.25)),
                #             levels=np.arange(1, max_util+0.25, 0.25),
                levels=levels
            )

            text = obs_i
            if obs_i == 'crosssection':
                text = 'dsg'
            ax.text(
                0.05, 0.95, text, transform=ax.transAxes,
                bbox=dict(facecolor='w', boxstyle='round', alpha=0.7), ha='left', va='top')
            if i == len(observables) - 1:
                ax.set_xlabel(fr'$\omega$ [MeV] ({subset_name})')
            if j == 0:
                ax.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
            ax.tick_params(direction='in')
            if i == 0:
                ax.set_title(subset_name)
    return fig, axes


def plot_utilities_all_observables(util_dict, nucleon, subset, omega, degrees, observables=None, max_util_dict=None,
                                   # xgrid=None, ygrid=None
                                   degrees_min=None, degrees_max=None, omega_min=None
                                   ):
    nucleons_all, observables_all, subsets_all = tuple(zip(*util_dict.keys()))
    nucleons = np.unique(nucleons_all)
    subsets = np.unique(subsets_all)
    if observables is None:
        observables = np.unique(observables_all)

    if nucleon not in nucleons:
        raise ValueError('nucleon must be in dict keys')
    if subset not in subsets:
        raise ValueError('subset must be in dict keys')

    n_omega = len(omega)
    n_degrees = len(degrees)

    ncols = 4
    fig, axes = plt.subplots(int(np.ceil(len(observables) / ncols)), ncols, figsize=(7, 5), sharex=True, sharey=True)
    for i, obs in enumerate(observables):
        ax = axes.ravel()[i]

        util_i = util_dict[nucleon, obs, subset]
        levels = None

        ax.contourf(
            omega, degrees, util_i.reshape(n_omega, n_degrees).T,
            # vmin=min_util, vmax=max_util,
            #                 locator=ticker.LogLocator(),
            #             levels=np.log(np.arange(1, max_util+0.25, 0.25)),
            #             levels=np.arange(1, max_util+0.25, 0.25),
            levels=levels
        )

        if max_util_dict is not None:
            max_utils = max_util_dict[nucleon, obs, subset]
            max_omega = max_utils['omega']
            max_degrees = max_utils['theta']
            ax.plot(
                max_omega, max_degrees, ls='', marker='o', c='r',
                fillstyle='none', markersize=4, markeredgewidth=0.6
            )

        # if xgrid is not None:
        #     [ax.axvline(xx, 0, 1, c='gray', lw=1) for xx in xgrid]
        # if ygrid is not None:
        #     [ax.axhline(yy, 0, 1, c='gray', lw=1) for yy in ygrid]
        fill_kwargs = dict(facecolor='none', edgecolor='grey', hatch='/')
        if degrees_min is not None:
            ax.fill_between([np.min(omega), np.max(omega)], [np.min(degrees), np.min(degrees)],
                            [degrees_min, degrees_min], **fill_kwargs)
        if degrees_max is not None:
            ax.fill_between([np.min(omega), np.max(omega)], [np.max(degrees), np.max(degrees)],
                            [degrees_max, degrees_max], **fill_kwargs)
        if omega_min is not None:
            d_max = degrees_max if degrees_max is not None else np.max(degrees)
            d_min = degrees_min if degrees_min is not None else np.min(degrees)
            ax.fill_between([np.min(omega), omega_min], [d_max, d_max],
                            [d_min, d_min], **fill_kwargs)

        text = obs
        if obs == 'crosssection':
            text = 'dsg'
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            bbox=dict(facecolor='w', boxstyle='round', alpha=0.7), ha='left', va='top')
        if i == len(observables) - 1:
            ax.set_xlabel(fr'$\omega$ [MeV]')
        if i % ncols == 0:
            ax.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
    for ax in axes.ravel():
        ax.tick_params(direction='in')
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    return fig, axes
