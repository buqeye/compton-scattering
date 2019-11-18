from numpy import sqrt, array

# Make sure these are right
mass_proton = 938.272
alpha_fine = 0.00729735256
hbarc = 197.326
fm2_to_nb = 1e7  # I think
pol_vec_names = ['alpha', 'beta', 'gammaE1E1', 'gammaM1M1', 'gammaE1M2', 'gammaM1E2']

# From Griesshammer et al. 2018 Table 1
proton_pol_vec_mean = array([10.65, 3.15, -1.1, 2.2, -0.4, 1.9])
proton_pol_vec_std = array([
    sqrt(0.35**2 + 0.2**2 + 0.3**2),
    sqrt(0.35**2 + 0.2**2 + 0.3**2),
    1.9,
    sqrt(0.5**2 + 0.6**2),
    0.6,
    0.5,
])
neutron_pol_vec_mean = array([11.55, 3.65, -4.0, 1.3, -0.1, 2.4])
neutron_pol_vec_std = array([
    sqrt(1.25**2 + 0.2**2 + 0.8**2),
    sqrt(1.25**2 + 0.2**2 + 0.8**2),
    1.9,
    sqrt(0.5**2 + 0.6**2),
    0.6,
    0.5,
])
