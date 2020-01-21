import numpy as np
from itertools import combinations, product
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from .constants import dsg_label, observables_unique, omega_lab_cusp, accuracy_levels


def ref_scale(omega, omega_pi, degrees, height, width=50, degrees_width=np.inf):
    if height == 1:
        return 1.
    return 1 / ((omega - omega_pi)**2/width**2 + degrees**2 / degrees_width**2 + 1/(height-1)) + 1


def compton_kernel(X, std, ell_omega, ell_degrees, noise_std=1e-7, degrees_zeros=None):
    kern_omega = RBF(ell_omega)
    kern_degrees = RBF(ell_degrees)

    deg = X[:, [1]]
    K_omega = kern_omega(X[:, [0]])
    K_degrees = kern_degrees(deg)

    if degrees_zeros is not None:
        degrees_zeros = np.atleast_1d(degrees_zeros)
        if degrees_zeros.ndim == 1:
            degrees_zeros = degrees_zeros[:, None]
        temp = np.linalg.solve(kern_degrees(degrees_zeros), kern_degrees(degrees_zeros, deg))
        K_degrees = K_degrees - kern_degrees(deg, degrees_zeros) @ temp

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
        kernel_kwargs = {(obs, nucleon): dict() for obs in observables_unique for nucleon in ['neutron', 'proton']}

    dsg_percent_error_p = accuracy_levels[level]['dsg_percent_error_p'] * scale_exp
    spin_absolute_error_p = accuracy_levels[level]['spin_absolute_error_p'] * scale_exp
    dsg_percent_error_n = accuracy_levels[level]['dsg_percent_error_n'] * scale_exp
    spin_absolute_error_n = accuracy_levels[level]['spin_absolute_error_n'] * scale_exp

    corr_identity = np.eye(X.shape[0])

    def create_cov_th(X, std=1, ls_omega=1e-8, ls_degrees=1e-8, noise_std=0, degrees_zeros=None, height=1):
        corr = compton_kernel(X, std, ls_omega, ls_degrees, noise_std=noise_std, degrees_zeros=degrees_zeros)

        if height is not None:
            ref = ref_scale(X[:, 0], omega_lab_cusp, X[:, 1], height)
        else:
            ref = 1.
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
            cov_trunc_p = Q_sum_p * create_cov_th(X, **kernel_kwargs[name, 'proton'])
            cov_trunc_n = Q_sum_n * create_cov_th(X, **kernel_kwargs[name, 'neutron'])
        else:
            cov_trunc_p = 0.
            cov_trunc_n = 0.

        return ComptonExperiment(
            name=name,
            cov_proton=cov_trunc_p + spin_absolute_error_p ** 2 * corr_identity,
            cov_neutron=cov_trunc_n + spin_absolute_error_n ** 2 * corr_identity,
        )

    if trunc:
        dsg_ref_p = dsg_pred_proton[:, None] * dsg_pred_proton
        dsg_ref_n = dsg_pred_neutron[:, None] * dsg_pred_neutron
        dsg_cov_trunc_p = dsg_ref_p * Q_sum_p * create_cov_th(X, **kernel_kwargs[dsg_label, 'proton'])
        dsg_cov_trunc_n = dsg_ref_n * Q_sum_n * create_cov_th(X, **kernel_kwargs[dsg_label, 'neutron'])
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
    bests_df['n pts'] = n_pts
    bests_df = bests_df.astype({'util': 'float64', 'n pts': 'int32'})
    bests_df['Shrinkage'] = np.exp(bests_df['util'])
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
        'n pts': n_pts
    })
    bests_df_flat['idx'] = bests_df_flat.astype({'idx': 'int32', 'n pts': 'int32'})['idx']
    bests_df_flat = bests_df_flat.sort_values(by=['idx'])
    bests_df_flat = bests_df_flat.reset_index()
    bests_df_flat = bests_df_flat.drop('index', axis=1)
    bests_df_flat['Shrinkage'] = np.exp(bests_df_flat['util'])
    if observable_order is not None:
        bests_df_flat['Observable'] = pd.Categorical(bests_df_flat['Observable'], observable_order, ordered=True)
    if subset_order is not None:
        bests_df_flat['Subset'] = pd.Categorical(bests_df_flat['Subset'], subset_order, ordered=True)
    return bests_df_flat
