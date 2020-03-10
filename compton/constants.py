from numpy import sqrt, array

# Make sure these are right
mass_proton = 938.272
mass_neutron = 939.565
mass_pion = 138
mass_delta = 1232
alpha_fine = 0.00729735256
hbarc = 197.326
fm2_to_nb = 1e7  # I think
pol_vec_names = ['alpha', 'beta', 'gammaE1E1', 'gammaM1M1', 'gammaE1M2', 'gammaM1E2']
pol_vec_trans_tex_names = [
    r'$\alpha+\beta$',
    r'$\alpha-\beta$',
    r'$\gamma_{0}$',
    r'$\gamma_{\pi}$',
    r'$\gamma_{E-}$',
    r'$\gamma_{M-}$',
]

dsg_label = r'$d\sigma$'

observables_name_map = {
    'crosssection': dsg_label,
    'dsg': dsg_label,
    '1X': r'$\Sigma_{1x}$',
    '1Z': r'$\Sigma_{1z}$',
    '2X': r'$\Sigma_{2x}$',
    '2Z': r'$\Sigma_{2z}$',
    '3': r'$\Sigma_{3}$',
    'Y': r'$\Sigma_{y}$',
    '3Y': r'$\Sigma_{3y}$',
    '3Yp': r"$\Sigma_{3y'}$",
    '1Xp': r"$\Sigma_{1x'}$",
    '1Zp': r"$\Sigma_{1z'}$",
    '2Xp': r"$\Sigma_{2x'}$",
    '2Zp': r"$\Sigma_{2z'}$",
}

observables_unique_raw = array(['dsg', '1X', '1Z', '2X', '2Z', '3', 'Y', '3Y', '3Yp', '1Xp', '1Zp', '2Xp', '2Zp'])
observables_unique = array([observables_name_map[name] for name in observables_unique_raw])

omega_lab_cusp = 149.95069260447417  # MeV

P_trans = array([
    [1,  1,  0,  0,  0,  0],
    [1, -1,  0,  0,  0,  0],
    [0,  0, -1, -1, -1, -1],
    [0,  0, -1,  1, -1,  1],
    [0,  0,  1,  0, -1,  0],
    [0,  0,  0,  1,  0, -1]
])

# From Griesshammer et al. 2018 Table 1
proton_pol_vec_mean = array([10.65, 3.15, -1.1, 2.2, -0.4, 1.9])
# proton_pol_vec_mean = array([11.55, 3.65, -3.9394769972873096, 1.3, -0.051868897981361184, 2.1885661242054266])
proton_pol_vec_std = array([
    sqrt(0.35**2 + 0.2**2 + 0.3**2),
    sqrt(0.35**2 + 0.2**2 + 0.3**2),
    1.9,
    sqrt(0.5**2 + 0.6**2),
    0.6,
    0.5,
])

# a+b, a-b, g_0, g_pi, g_e-, g_m-
proton_pol_vec_trans_mean = array([13.8, 7.6, -1.0, 8.0, -1.1 - (-0.4), 2.2 - 1.9])
# See table 1 of Griesshammer 2016
proton_pol_vec_trans_std = array([
    0.4,  # a+b
    0.9,  # a-b
    sqrt(0.1**2 + 0.1**2),  # g_0
    1.8,  # g_pi
    sqrt(1.9**2 + 0.6**2),  # g_e-
    sqrt(0.5**2 + 0.6**2 + 0.5**2)  # g_m-
])

# neutron_pol_vec_mean = array([11.55, 3.65, -4.0, 1.3, -0.1, 2.4])
neutron_pol_vec_mean = array([11.55, 3.65, -3.9394769972873096, 1.3, -0.051868897981361184, 2.1885661242054266])
# {alpha -> 11.55`, beta -> 3.65`, gammaE1E1 -> -3.9394769972873096`,
#  gammaM1M1 -> 1.3`, gammaE1M2 -> -0.051868897981361184`,
#  gammaM1E2 -> 2.1885661242054266`}
neutron_pol_vec_std = array([
    sqrt(1.25**2 + 0.2**2 + 0.8**2),
    sqrt(1.25**2 + 0.2**2 + 0.8**2),
    1.9,
    sqrt(0.5**2 + 0.6**2),
    0.6,
    0.5,
])

neutron_pol_vec_trans_mean = P_trans @ neutron_pol_vec_mean  # Get real values!!
neutron_pol_vec_trans_std = proton_pol_vec_trans_std.copy()  # Get real values!!

accuracy_levels = {
        "standard": dict(
            dsg_percent_error_p=5,
            spin_absolute_error_p=0.1,
            dsg_percent_error_n=5,
            spin_absolute_error_n=0.1,
        ),
        "doable": dict(
            dsg_percent_error_p=4,
            spin_absolute_error_p=0.06,
            dsg_percent_error_n=4,
            spin_absolute_error_n=0.06,
        ),
        "aspirational": dict(
            dsg_percent_error_p=3,
            spin_absolute_error_p=0.03,
            dsg_percent_error_n=3,
            spin_absolute_error_n=0.03,
        )
    }


class PrecisionLevels:
    standard = 'standard'
    doable = 'doable'
    aspirational = 'aspirational'


class DesignLabels:

    n_pts = r'$\#$ Points'
    omega = 'omega'
    theta = 'theta'
    util = 'util'
    subset = 'Subset'
    shrinkage = 'Shrinkage'
    fvr = 'FVR'
    observable = 'Observable'
    nucleon = 'Nucleon'
    index = 'idx'
    indices = 'idxs'

    neutron = 'Neutron'
    proton = 'Proton'


nucleon_names = [DesignLabels.neutron, DesignLabels.proton]
