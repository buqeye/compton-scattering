import numpy as np
from numpy import cos, deg2rad
from .constants import alpha_fine, hbarc, fm2_to_nb, mass_proton


def omega_cm_from_lab(omega_lab, mass):
    return omega_lab * np.sqrt(mass / (mass + 2 * omega_lab))


def compute_cos0_lab(theta_lab, degrees=True):
    if degrees:
        theta_lab = np.deg2rad(theta_lab)
    return np.cos(theta_lab)


def omega_outgoing_lab(omega_lab, mass, cos0_lab):
    return omega_lab / (1. + omega_lab * (1 - cos0_lab) / mass)


def theta_cm_from_lab(omega_lab, mass, cos0_lab):
    sin0_lab = np.sqrt(1 - cos0_lab ** 2)
    num = sin0_lab / gamma_boost(omega_lab, mass)
    den = cos0_lab - beta_boost(omega_lab, mass)
    return np.rad2deg(np.arctan2(num, den))


def cos0_cm_from_lab(omega_lab, mass, cos0_lab):
    sin0_lab = np.sqrt(1 - cos0_lab ** 2)
    gamma = gamma_boost(omega_lab, mass)
    leg = gamma * (cos0_lab - beta_boost(omega_lab, mass))
    hyp = np.sqrt(leg ** 2 + sin0_lab ** 2)
    return leg / hyp


def momentum_transfer_cm(omega_cm, cos0_cm):
    return omega_cm * np.sqrt(2 * (1 - cos0_cm))


def momentum_transfer_lab(omega_lab, mass, cos0_lab):
    omega_prime = omega_outgoing_lab(omega_lab, mass, cos0_lab)
    sin0_lab = np.sqrt(1 - cos0_lab ** 2)
    q2 = (omega_lab - omega_prime * cos0_lab) ** 2 + (omega_prime * sin0_lab) ** 2
    return np.sqrt(q2)


def beta_boost(omega_lab, mass):
    return omega_lab / np.sqrt(omega_lab ** 2 + mass ** 2)


def gamma_boost(omega_lab, mass):
    beta = beta_boost(omega_lab, mass)
    return 1. / np.sqrt(1. - beta ** 2)


def dsg_proton_low_energy(cos0):
    """The low-energy limit of the proton's differential cross section

    """
    return (alpha_fine * hbarc / mass_proton)**2 * fm2_to_nb * \
        (1 + cos0 ** 2) / 2


def sigma3_low_energy(cos0):
    return (cos0 ** 2 - 1) / (cos0 ** 2 + 1)
