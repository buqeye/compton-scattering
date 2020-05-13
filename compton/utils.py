import numpy as np
from itertools import combinations, product
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from .constants import dsg_label, observables_unique, omega_lab_cusp, accuracy_levels, nucleon_names, DesignLabels


def ref_scale(omega, omega_pi, degrees, height, width=50, degrees_width=np.inf):
    if height == 1:
        return 1.
    return 1 / ((omega - omega_pi)**2/width**2 + degrees**2 / degrees_width**2 + 1/(height-1)) + 1


def compton_kernel(
        X, std, ell_omega, ell_degrees, noise_std=1e-7, degrees_zeros=None, omega_zeros=None,
        degrees_deriv_zeros=None, omega_deriv_zeros=None,
):

    deg = t = X[:, [1]]
    omega = w = X[:, [0]]

    import gptools

    kern_omega = gptools.SquaredExponentialKernel(
        initial_params=[1, ell_omega], fixed_params=[True, True])
    kern_theta = gptools.SquaredExponentialKernel(
        initial_params=[1, ell_degrees], fixed_params=[True, True])
    gp_omega = gptools.GaussianProcess(kern_omega)
    gp_theta = gptools.GaussianProcess(kern_theta)

    if omega_zeros is not None or omega_deriv_zeros is not None:
        w_z = []
        n_w = []

        if omega_zeros is not None:
            w_z.append(omega_zeros)
            n_w.append(np.zeros(len(omega_zeros)))
        if omega_deriv_zeros is not None:
            w_z.append(omega_deriv_zeros)
            n_w.append(np.ones(len(omega_deriv_zeros)))
        w_z = np.concatenate(w_z)[:, None]
        n_w = np.concatenate(n_w)

        gp_omega.add_data(w_z, np.zeros(w_z.shape[0]), n=n_w)
        _, K_omega = gp_omega.predict(w, np.zeros(w.shape[0]), return_cov=True)
    else:
        K_omega = gp_omega.compute_Kij(w, w, np.zeros(w.shape[0]), np.zeros(w.shape[0]))

    if degrees_zeros is not None or degrees_deriv_zeros is not None:
        t_z = []
        n_t = []

        if degrees_zeros is not None:
            t_z.append(degrees_zeros)
            n_t.append(np.zeros(len(degrees_zeros)))
        if degrees_deriv_zeros is not None:
            t_z.append(degrees_deriv_zeros)
            n_t.append(np.ones(len(degrees_deriv_zeros)))
        t_z = np.concatenate(t_z)[:, None]
        n_t = np.concatenate(n_t)

        gp_theta.add_data(t_z, np.zeros(t_z.shape[0]), n=n_t)
        _, K_degrees = gp_theta.predict(t, np.zeros(t.shape[0]), return_cov=True)
    else:
        K_degrees = gp_theta.compute_Kij(t, t, np.zeros(t.shape[0]), np.zeros(t.shape[0]))

    # kern_omega = RBF(ell_omega)
    # kern_degrees = RBF(ell_degrees)

    # K_omega = kern_omega(omega)
    # K_degrees = kern_degrees(deg)
    #
    # # Create conditional kernels if observables are known to vanish at certain locations
    # if omega_zeros is not None:
    #     omega_zeros = np.atleast_1d(omega_zeros)
    #     if omega_zeros.ndim == 1:
    #         omega_zeros = omega_zeros[:, None]
    #     temp_omega = np.linalg.solve(kern_omega(omega_zeros), kern_omega(omega_zeros, omega))
    #     K_omega = K_omega - kern_omega(omega, omega_zeros) @ temp_omega
    #
    # if degrees_zeros is not None:
    #     degrees_zeros = np.atleast_1d(degrees_zeros)
    #     if degrees_zeros.ndim == 1:
    #         degrees_zeros = degrees_zeros[:, None]
    #     temp = np.linalg.solve(kern_degrees(degrees_zeros), kern_degrees(degrees_zeros, deg))
    #     K_degrees = K_degrees - kern_degrees(deg, degrees_zeros) @ temp

    K = std ** 2 * K_omega * K_degrees
    K += noise_std ** 2 * np.eye(K.shape[0])
    return K


class ComptonExperiment:

    def __init__(self, name, cov_proton, cov_neutron):
        self.name = name
        self.cov_proton = cov_proton
        self.cov_neutron = cov_neutron


def create_experiment_infos(X, level, dsg_pred_proton, dsg_pred_neutron, Q_sum_p, Q_sum_n, trunc=True, scale_exp=1.,
                            kernel_kwargs=None):
    R"""

    Parameters
    ----------
    X
    level : str
        One of 'standard', 'doable' or 'aspirational'
    dsg_pred_proton
    dsg_pred_neutron
    Q_sum_p
    Q_sum_n
    trunc
    scale_exp
    kernel_kwargs

    Returns
    -------

    """

    # level = "standard today"
    if kernel_kwargs is None:
        kernel_kwargs = {(obs, nucleon): dict() for obs in observables_unique for nucleon in nucleon_names}

    dsg_percent_error_p = accuracy_levels[level]['dsg_percent_error_p'] * scale_exp
    spin_absolute_error_p = accuracy_levels[level]['spin_absolute_error_p'] * scale_exp
    dsg_percent_error_n = accuracy_levels[level]['dsg_percent_error_n'] * scale_exp
    spin_absolute_error_n = accuracy_levels[level]['spin_absolute_error_n'] * scale_exp

    corr_identity = np.eye(X.shape[0])

    def create_cov_th(
            X, std=1, ls_omega=1e-8, ls_degrees=1e-8, noise_std=0, degrees_zeros=None, omega_zeros=None,
            degrees_deriv_zeros=None, omega_deriv_zeros=None,
            height=1, ref=1, width=50, degrees_width=np.inf,
    ):
        corr = compton_kernel(
            X, std, ls_omega, ls_degrees, noise_std=noise_std, degrees_zeros=degrees_zeros, omega_zeros=omega_zeros,
            degrees_deriv_zeros=degrees_deriv_zeros, omega_deriv_zeros=omega_deriv_zeros,
        )

        if height is not None:
            ref = ref * ref_scale(
                X[:, 0], omega_pi=omega_lab_cusp, degrees=X[:, 1], height=height, width=width,
                degrees_width=degrees_width
            )
        else:
            ref = 1. * ref
        ref = np.atleast_1d(ref)
        cov = ref[:, None] * ref * corr

        # sd = np.array(sd)
        # if sd.ndim == 0 or sd.ndim == 2:
        #     cov = sd ** 2 * corr
        # elif sd.ndim == 1:
        #     cov = sd[:, None] * sd * corr
        # else:
        #     raise ValueError('sd must be 0, 1 or 2d')

        # noise = np.array(noise)
        # if noise.ndim == 0:
        #     cov += noise ** 2 * np.eye(cov.shape[0])
        # elif noise.ndim == 1:
        #     cov += np.diag(noise ** 2)
        # elif noise.ndim == 2:
        #     cov += noise ** 2
        # else:
        #     raise ValueError('noise must be 0, 1 or 2d')

        return cov

    def create_spin_expt(name):
        if trunc:
            cov_trunc_p = Q_sum_p * create_cov_th(X, **kernel_kwargs[name, DesignLabels.proton])
            cov_trunc_n = Q_sum_n * create_cov_th(X, **kernel_kwargs[name, DesignLabels.neutron])

        else:
            cov_trunc_p = 0.
            cov_trunc_n = 0.

        return ComptonExperiment(
            name=name,
            cov_proton=cov_trunc_p + spin_absolute_error_p ** 2 * corr_identity,
            cov_neutron=cov_trunc_n + spin_absolute_error_n ** 2 * corr_identity,
        )

    if trunc:
        # dsg_ref_p = dsg_pred_proton[:, None] * dsg_pred_proton
        # dsg_ref_n = dsg_pred_neutron[:, None] * dsg_pred_neutron
        # Taken into account via ref scale now.
        dsg_ref_p = 1.
        dsg_ref_n = 1.
        dsg_cov_trunc_p = dsg_ref_p * Q_sum_p * create_cov_th(X, **kernel_kwargs[dsg_label, DesignLabels.proton])
        dsg_cov_trunc_n = dsg_ref_n * Q_sum_n * create_cov_th(X, **kernel_kwargs[dsg_label, DesignLabels.neutron])
    else:
        dsg_cov_trunc_p = 0
        dsg_cov_trunc_n = 0

    dsg_info = ComptonExperiment(
        name=dsg_label,
        cov_proton=dsg_cov_trunc_p + np.diag(dsg_percent_error_p / 100. * dsg_pred_proton) ** 2,
        cov_neutron=dsg_cov_trunc_n + np.diag(dsg_percent_error_n / 100. * dsg_pred_neutron) ** 2,
    )
    expts_info = {
        name: create_spin_expt(name)
        for name in observables_unique[1:]
    }
    expts_info[dsg_label] = dsg_info
    return expts_info


def compute_all_1pt_utilities(obs_dict, subsets=None):
    util_dict = {}

    observables_all, nucleons_all, orders_all, type_all = tuple(zip(*obs_dict.keys()))
    nucleons = np.unique(nucleons_all)
    observables = np.unique(observables_all)

    if subsets is None:
        subsets = {'all': None}

    for i, obs_i in enumerate(observables):
        for j, (subset_name, subset) in enumerate(subsets.items()):
            for nucleon in nucleons:
                order_i = 4
                compton_i = obs_dict[obs_i, nucleon, order_i, 'linear']

                util_i = np.zeros(compton_i.n_data, dtype=float)
                for n in range(compton_i.n_data):
                    util_i[n] = compton_i.utility_linear([n], p_idx=subset)
                util_dict[nucleon, obs_i, subset_name] = util_i
    return util_dict


def compute_max_utilities(obs_dict, subsets, omega, degrees, n_degrees=1, verbose=False):
    max_util_dict = {}

    observables_all, nucleons_all, orders_all, type_all = tuple(zip(*obs_dict.keys()))
    nucleons = np.unique(nucleons_all)
    observables = np.unique(observables_all)

    for nucleon in nucleons:
        for subset_name, subset in subsets.items():
            bests = {obs: dict(util=-np.inf) for obs in observables}
            if verbose:
                print(subset_name)
            for aa, angles in enumerate(combinations(degrees, n_degrees)):
                if aa % 100 == 0 and verbose:
                    print(aa)
                for energy in omega:
                    for obs in observables:
                        order = 4
                        compton_obs = obs_dict[obs, nucleon, order, 'linear']
                        mask = (compton_obs.omega_lab == energy) & np.isin(compton_obs.degrees_lab, angles)
                        idxs = np.where(mask)[0]

                        current_util = compton_obs.utility_linear(idxs, p_idx=subset)
                        if current_util > bests[obs]['util']:
                            bests_i = bests[obs]
                            bests_i['util'] = current_util
                            bests_i['idxs'] = idxs
                            bests_i['omega'] = compton_obs.omega_lab[idxs]
                            bests_i['theta'] = compton_obs.degrees_lab[idxs]
                            max_util_dict[nucleon, obs, subset_name] = bests_i
    return max_util_dict


def convert_max_utilities_to_dataframe(max_utilities, observable_order=None, subset_order=None):
    bests_df = pd.DataFrame.from_dict(max_utilities).T
    bests_df.index.names = 'Nucleon', 'Observable', 'Subset'
    bests_df = bests_df.reset_index()
    # print(bests_df)
    n_pts = len(bests_df.loc[0, 'idxs'])
    n_pts_label = r'$\#$ Points'
    bests_df[n_pts_label] = n_pts
    # bests_df = bests_df.astype({'util': 'float64', n_pts_label: 'int32'})
    bests_df = bests_df.astype({'util': 'float64', n_pts_label: int})
    n_pts_ints = np.sort(bests_df[n_pts_label].unique())
    n_pts_strs = []
    for n in n_pts_ints:
        if n > 1:
            n_pts_strs.append(str(n) + ' Points')
        else:
            n_pts_strs.append(str(n) + ' Point')
    bests_df[n_pts_label] = bests_df[n_pts_label].replace(dict(zip(n_pts_ints, n_pts_strs)))
    bests_df[n_pts_label] = pd.Categorical(bests_df[n_pts_label], n_pts_strs[::-1], ordered=True)

    # one_pt_mask = bests_df[n_pts_label] == '1'
    # bests_df.loc[one_pt_mask, n_pts_label] += ' Points'
    # bests_df.loc[~one_pt_mask, n_pts_label] += ' Points'
    bests_df['Shrinkage'] = np.exp(bests_df['util'])
    bests_df['FVR'] = 1 - 1. / bests_df['Shrinkage']
    if observable_order is not None:
        bests_df['Observable'] = pd.Categorical(bests_df['Observable'], observable_order, ordered=True)
    if subset_order is not None:
        bests_df['Subset'] = pd.Categorical(bests_df['Subset'], subset_order, ordered=True)
    return bests_df


def convert_max_utilities_to_flat_dataframe(max_utilities, observable_order=None, subset_order=None):
    idxs_flat = []
    omega_flat = []
    theta_flat = []
    nucleon_flat = []
    obs_flat = []
    subset_flat = []
    util_flat = []

    n_pts = None

    n_pts_label = r'$\#$ Points'

    for (nucleon, obs, subset), best in max_utilities.items():
        if n_pts is None:
            n_pts = len(list(best['idxs']))
        idxs_flat = idxs_flat + list(best['idxs'])
        omega_flat = omega_flat + list(best['omega'])
        theta_flat = theta_flat + list(best['theta'])
        nucleon_flat = nucleon_flat + [nucleon for _ in range(len(best['idxs']))]
        obs_flat = obs_flat + [obs for _ in range(len(best['idxs']))]
        subset_flat = subset_flat + [subset for _ in range(len(best['idxs']))]
        util_flat = util_flat + [best['util'] for _ in range(len(best['idxs']))]

    bests_df_flat = pd.DataFrame({
        'idx': idxs_flat, 'omega': omega_flat, 'theta': theta_flat,
        'Nucleon': nucleon_flat, 'Observable': obs_flat, 'util': util_flat, 'Subset': subset_flat,
        n_pts_label: n_pts
    })
    bests_df_flat = bests_df_flat.astype({'idx': 'int32', n_pts_label: int})
    # bests_df_flat['idx'] = bests_df_flat.astype({'idx': 'int32', n_pts_label: 'int32'})['idx']
    n_pts_ints = bests_df_flat[n_pts_label].unique()
    n_pts_strs = []
    for n in n_pts_ints:
        if n > 1:
            n_pts_strs.append(str(n) + ' Points')
        else:
            n_pts_strs.append(str(n) + ' Point')
    # one_pt_mask = bests_df_flat[n_pts_label] == '1'
    # bests_df_flat.loc[one_pt_mask, n_pts_label] += ' Points'
    # bests_df_flat.loc[~one_pt_mask, n_pts_label] += ' Points'
    bests_df_flat[n_pts_label] = bests_df_flat[n_pts_label].replace(dict(zip(n_pts_ints, n_pts_strs)))
    bests_df_flat[n_pts_label] = pd.Categorical(bests_df_flat[n_pts_label], n_pts_strs[::-1], ordered=True)

    bests_df_flat = bests_df_flat.sort_values(by=['idx', n_pts_label])
    bests_df_flat = bests_df_flat.reset_index()
    bests_df_flat = bests_df_flat.drop('index', axis=1)
    bests_df_flat['Shrinkage'] = np.exp(bests_df_flat['util'])
    bests_df_flat['FVR'] = 1 - 1. / bests_df_flat['Shrinkage']  # Fraction of uncertainty removed
    if observable_order is not None:
        bests_df_flat['Observable'] = pd.Categorical(bests_df_flat['Observable'], observable_order, ordered=True)
    if subset_order is not None:
        bests_df_flat['Subset'] = pd.Categorical(bests_df_flat['Subset'], subset_order, ordered=True)
    return bests_df_flat
