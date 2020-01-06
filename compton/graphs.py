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


def plot_allowed_experimental_region(omega, degrees, degrees_min=None, degrees_max=None, omega_min=None, ax=None):
    if ax is None:
        ax = plt.gca()

    fill_kwargs = dict(facecolor='none', edgecolor='grey', hatch='/', linewidth=0.0)
    if degrees_min is not None:
        ax.fill_between([np.min(omega), np.max(omega)], [np.min(degrees), np.min(degrees)],
                        [degrees_min, degrees_min], **fill_kwargs)
        ax.plot([omega_min or np.min(omega), np.max(omega)], [degrees_min, degrees_min], color='gray', lw=0.5)
    if degrees_max is not None:
        ax.fill_between([np.min(omega), np.max(omega)], [np.max(degrees), np.max(degrees)],
                        [degrees_max, degrees_max], **fill_kwargs)
        ax.plot([omega_min or np.min(omega), np.max(omega)], [degrees_max, degrees_max], color='gray', lw=0.5)
    if omega_min is not None:
        d_max = degrees_max if degrees_max is not None else np.max(degrees)
        d_min = degrees_min if degrees_min is not None else np.min(degrees)
        ax.fill_between([np.min(omega), omega_min], [d_max, d_max],
                        [d_min, d_min], **fill_kwargs)
        ax.plot([omega_min, omega_min], [degrees_min or np.min(degrees), degrees_max or np.max(degrees)],
                color='gray', lw=0.5)
    return ax


def plot_utilities_all_observables(
        util_dict, nucleon, subset, omega, degrees, observables=None, max_util_dict=None,
        degrees_min=None, degrees_max=None, omega_min=None, axes=None, **kwargs
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
    nrows = int(np.ceil(len(observables) / ncols))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 3.5), sharex=True, sharey=True)
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
            levels=levels,
            **kwargs
        )

        if max_util_dict is not None:
            max_utils = max_util_dict[nucleon, obs, subset]
            max_omega = max_utils['omega']
            max_degrees = max_utils['theta']
            ax.plot(
                max_omega, max_degrees, ls='', marker='o', c='k',
                fillstyle='none', markersize=4, markeredgewidth=0.6
            )

        plot_allowed_experimental_region(
            omega, degrees, degrees_min=degrees_min, degrees_max=degrees_max, omega_min=omega_min, ax=ax
        )

        text = obs
        if obs == 'crosssection':
            text = 'dsg'
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            bbox=dict(facecolor='w', boxstyle='round', alpha=0.7), ha='left', va='top')
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(fr'$\omega$ [MeV]')
        if i % ncols == 0:
            ax.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
    for ax in axes.ravel():
        ax.tick_params(direction='in')
    fig = plt.gcf()
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    return fig, axes


def plot_comparison_subsets_and_truncation(
        util_dict_no_trunc, util_dict_with_trunc, nucleon, observable, omega, degrees, max_util_dict_no_trunc=None,
        max_util_dict_with_trunc=None, degrees_min=None, degrees_max=None, omega_min=None, axes=None,
        subsets=None, **kwargs):
    nucleons_all, observables_all, subsets_all = tuple(zip(*util_dict_no_trunc.keys()))
    nucleons = np.unique(nucleons_all)
    if subsets is None:
        subsets = np.unique(subsets_all)
    observables = np.unique(observables_all)

    n_omega = len(omega)
    n_degrees = len(degrees)

    if nucleon not in nucleons:
        raise ValueError('nucleon must be in dict keys')
    if observable not in observables:
        raise ValueError('observable must be in dict keys')

    nrows = 2
    ncols = len(subsets)
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 3.5), sharex=True, sharey=True)

    for i, subset in enumerate(subsets):
        ax = axes[0, i]
        ax_trunc = axes[1, i]

        util_no_trunc = util_dict_no_trunc[nucleon, observable, subset]
        util_with_trunc = util_dict_with_trunc[nucleon, observable, subset]

        ax.contourf(
            omega, degrees, util_no_trunc.reshape(n_omega, n_degrees).T,
            **kwargs
        )
        ax_trunc.contourf(
            omega, degrees, util_with_trunc.reshape(n_omega, n_degrees).T,
            **kwargs
        )

        if max_util_dict_no_trunc is not None:
            max_utils = max_util_dict_no_trunc[nucleon, observable, subset]
            best_omega = max_utils['omega']
            best_degrees = max_utils['theta']
            ax.plot(
                best_omega, best_degrees, ls='', marker='o', c='k',
                fillstyle='none', markersize=4, markeredgewidth=0.6
            )

        if max_util_dict_with_trunc is not None:
            max_utils = max_util_dict_with_trunc[nucleon, observable, subset]
            best_omega = max_utils['omega']
            best_degrees = max_utils['theta']
            ax_trunc.plot(
                best_omega, best_degrees, ls='', marker='o', c='k',
                fillstyle='none', markersize=4, markeredgewidth=0.6
            )

        plot_allowed_experimental_region(
                omega, degrees, degrees_min=degrees_min, degrees_max=degrees_max, omega_min=omega_min, ax=ax
        )
        plot_allowed_experimental_region(
            omega, degrees, degrees_min=degrees_min, degrees_max=degrees_max, omega_min=omega_min, ax=ax_trunc
        )

        # text_no_trunc = r'$\xcancel{\delta y}$'
        text_no_trunc = r'$\delta y$'
        text_with_trunc = r'$\delta y$'
        ax.text(
            0.05, 0.95, text_no_trunc, transform=ax.transAxes,
            bbox=dict(facecolor='w', boxstyle='round', alpha=0.7, hatch='xxx', edgecolor='r'), ha='left', va='top'
        )
        ax_trunc.text(
            0.05, 0.95, text_with_trunc, transform=ax_trunc.transAxes,
            bbox=dict(facecolor='w', boxstyle='round', alpha=0.7, edgecolor='g'), ha='left', va='top'
        )

        ax_trunc.set_xlabel(fr'$\omega$ [MeV]')
        if i % ncols == 0:
            ax.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
            ax_trunc.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
        ax.set_title(subset)
    for ax in axes.ravel():
        ax.tick_params(direction='in')
    fig = plt.gcf()
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    return fig, axes


def plot_comparison_subsets_for_observables(
        util_dict, omega, degrees, max_util_dict=None,
        degrees_min=None, degrees_max=None, omega_min=None, axes=None,
        subsets=None, observables=None, cmap_p=None, cmap_n=None, **kwargs):
    nucleons_all, observables_all, subsets_all = tuple(zip(*util_dict.keys()))
    nucleons = np.unique(nucleons_all)
    if subsets is None:
        subsets = np.unique(subsets_all)
    if observables is None:
        observables = np.unique(observables_all)

    n_omega = len(omega)
    n_degrees = len(degrees)

    nrows = len(observables)
    ncols = 2*len(subsets)
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 8), sharex=True, sharey=True)

    for i, obs in enumerate(observables):
        for j, subset in enumerate(subsets):
            for n, nucleon in enumerate(['proton', 'neutron']):
                ax = axes[i, j+3*n]

                util_i = util_dict[nucleon, obs, subset]

                ax.contourf(
                    omega, degrees, util_i.reshape(n_omega, n_degrees).T,
                    cmap=cmap_p if nucleon == 'proton' else cmap_n,
                    **kwargs
                )

                if max_util_dict is not None:
                    max_utils = max_util_dict[nucleon, obs, subset]
                    max_omega = max_utils['omega']
                    max_degrees = max_utils['theta']
                    ax.plot(
                        max_omega, max_degrees, ls='', marker='o', c='k',
                        fillstyle='none', markersize=4, markeredgewidth=0.6
                    )

                plot_allowed_experimental_region(
                    omega, degrees, degrees_min=degrees_min, degrees_max=degrees_max, omega_min=omega_min, ax=ax
                )

                text = obs
                if obs == 'crosssection':
                    text = 'dsg'
                ax.text(
                    0.05, 0.95, text, transform=ax.transAxes,
                    bbox=dict(facecolor='w', boxstyle='round', alpha=0.7), ha='left', va='top')

                if i == 0:
                    ax.set_title(subset)
                if i >= (nrows - 1):
                    ax.set_xlabel(fr'$\omega$ [MeV]')
                if j+3*n == 0:
                    ax.set_ylabel(r'$\theta_{\rm lab}$ [deg]')
    for ax in axes.ravel():
        ax.tick_params(direction='in')
    fig = plt.gcf()
    fig.tight_layout(h_pad=0.3, w_pad=0.28)
    return fig, axes
