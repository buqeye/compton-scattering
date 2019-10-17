import numpy as np
from numpy import pi


def breit_amplitudes(params, omega, z, A=None):
    R"""The low-energy expansion of the non-Born parts of the Breit-frame amplitudes in terms of the polarisabilities.

    This also works for the cm frame, apparently. See Eq. (B.2) from Griesshammer 2018.

    Parameters
    ----------
    params : ndarray
        The polarizabilities [a_E1, b_M1, g_E1E1, g_M1M1, g_E1M2, g_M1E2]
    omega : ndarray
        The frequency in the appropriate frame
    z :
        cos(theta) in the appropriate frame
    A : ndarray
        The amplitude matrix in which to store the output. Creates a new array if `None` is given.

    Returns
    -------

    """
    aE1, bM1, gE1E1, gM1M1, gE1M2, gM1E2 = params
    if A is None:
        A = np.empty((6, len(omega), len(z)))
    omega = omega[:, None]

    A[0] = + (aE1 + z * bM1) * omega ** 2
    A[1] = - bM1 * omega ** 2
    A[2] = - (gE1E1 + gE1M2 + z * (gM1M1 + gM1E2)) * omega ** 3
    A[3] = + (gM1E2 - gM1M1) * omega ** 3
    A[4] = + gM1M1 * omega ** 3
    A[5] = + gE1M2 * omega ** 3

    A *= 4 * pi
    return A
