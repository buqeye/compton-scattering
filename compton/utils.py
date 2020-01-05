import numpy as np
from itertools import combinations, product
import pandas as pd


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


def compute_max_utilities(obs_dict, subsets, omega, degrees, n_degrees=1):
    max_util_dict = {}

    observables_all, nucleons_all, orders_all, type_all = tuple(zip(*obs_dict.keys()))
    nucleons = np.unique(nucleons_all)
    observables = np.unique(observables_all)

    for nucleon in nucleons:
        for subset_name, subset in subsets.items():
            bests = {obs: dict(util=-np.inf) for obs in observables}
            print(subset_name)
            for aa, angles in enumerate(combinations(degrees, n_degrees)):
                if aa % 100 == 0:
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


def convert_max_utilities_to_dataframe(max_utilities):
    bests_df = pd.DataFrame.from_dict(max_utilities).T
    bests_df.index.names = 'nucleon', 'observable', 'subset'
    bests_df = bests_df.reset_index()
    bests_df['util'] = bests_df.astype({'util': 'float64'})['util']
    bests_df['shrinkage'] = np.exp(bests_df['util'])
    return bests_df


def convert_max_utilities_to_flat_dataframe(max_utilities):
    idxs_flat = []
    omega_flat = []
    theta_flat = []
    nucleon_flat = []
    obs_flat = []
    subset_flat = []
    util_flat = []

    for (nucleon, obs, subset), best in max_utilities.items():
        idxs_flat = idxs_flat + list(best['idxs'])
        omega_flat = omega_flat + list(best['omega'])
        theta_flat = theta_flat + list(best['theta'])
        nucleon_flat = nucleon_flat + [nucleon for _ in range(len(best['idxs']))]
        obs_flat = obs_flat + [obs for _ in range(len(best['idxs']))]
        subset_flat = subset_flat + [subset for _ in range(len(best['idxs']))]
        util_flat = util_flat + [best['util'] for _ in range(len(best['idxs']))]

    bests_df_flat = pd.DataFrame(
        {'idx': idxs_flat, 'omega': omega_flat, 'theta': theta_flat,
         'nucleon': nucleon_flat, 'observable': obs_flat, 'util': util_flat, 'subset': subset_flat}
    )
    bests_df_flat['idx'] = bests_df_flat.astype({'idx': 'int32'})['idx']
    bests_df_flat = bests_df_flat.sort_values(by=['idx'])
    bests_df_flat = bests_df_flat.reset_index()
    bests_df_flat = bests_df_flat.drop('index', axis=1)
    bests_df_flat['shrinkage'] = np.exp(bests_df_flat['util'])
    return bests_df_flat
