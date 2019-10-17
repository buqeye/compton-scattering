import numpy as np
from numpy import pi


def breit_amplitudes(params, omega, z, A=None, dA=None):
    R"""The low-energy expansion of the non-Born parts of the Breit-frame amplitudes in terms of the polarisabilities.

    This also works for the cm frame, apparently. See Eq. (B.2) from Griesshammer 2018.

    Parameters
    ----------
    params : ndarray, shape = (p,)
        The polarizabilities [a_E1, b_M1, g_E1E1, g_M1M1, g_E1M2, g_M1E2]
    omega : ndarray, shape = (n_omega,)
        The frequency in the appropriate frame
    z : ndarray, shape = (n_z,)
        cos(theta) in the appropriate frame
    A : ndarray, shape = (6, n_omega, n_z)
        The amplitude matrix in which to store the output. Creates a new array if `None` is given.
    dA : ndarray, shape = (p, 6, n_omega, n_z)
        The array in which to store the gradient of the amplitude wrt the polarizabilities.

    Returns
    -------
    A : ndarray, shape = (6, n_omega, n_z)
        The amplitudes
    dA : ndarray, shape = (p, 6, n_omega, n_z)
        The gradient of the amplitudes wrt the polarizabilities. Only returned if dA is not None.
    """
    aE1, bM1, gE1E1, gM1M1, gE1M2, gM1E2 = params
    if A is None:
        A = np.empty((6, len(omega), len(z)))
    omega = omega[:, None]

    A[0] = + omega ** 2 * (aE1 + z * bM1)
    A[1] = - omega ** 2 * bM1
    A[2] = - omega ** 3 * (gE1E1 + gE1M2 + z * (gM1M1 + gM1E2))
    A[3] = + omega ** 3 * (gM1E2 - gM1M1)
    A[4] = + omega ** 3 * gM1M1
    A[5] = + omega ** 3 * gE1M2
    A *= 4 * pi

    if dA is not None:
        dA[...] = 0.  # Reset all values
        # dA / daE1
        dA[0, 0] = + omega ** 2
        # dA / dbM1
        dA[1, 0] = + omega ** 2 * z
        dA[1, 1] = - omega ** 2
        # dA / dgE1E1
        dA[2, 2] = - omega ** 3
        # dA / dgM1M1
        dA[3, 2] = - omega ** 3 * z
        dA[3, 3] = - omega ** 3
        dA[3, 4] = + omega ** 3
        # dA / dgE1M2
        dA[4, 2] = - omega ** 3
        dA[4, 5] = + omega ** 3
        # dA / dgM1E2
        dA[5, 2] = - omega ** 3 * z
        dA[5, 3] = + omega ** 3

        dA *= 4 * pi
        return A, dA

    return A
