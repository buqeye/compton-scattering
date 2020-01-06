import numpy as np
from numpy.linalg import slogdet, solve
from numpy import log, pi
import pandas as pd
from scipy.special import expit
from .constants import mass_pion
from .kinematics import momentum_transfer_cm, cos0_cm_from_lab, omega_cm_from_lab
from sklearn.gaussian_process.kernels import RBF


def order_transition(n, n_inf, omega):
    return n + (n_inf - n) * expit((omega-190)/20)


def expansion_parameter(X, breakdown):
    X = np.atleast_2d(X)
    return np.squeeze((X[:, 0] + mass_pion) / breakdown)


def expansion_parameter_transfer_cm(X, breakdown, mass):
    X = np.atleast_2d(X)
    omega_lab, cos0_lab = X.T
    cos0_lab = np.cos(np.deg2rad(cos0_lab))
    omega_cm = omega_cm_from_lab(omega_lab, mass=mass)
    cos0_cm = cos0_cm_from_lab(omega_lab, mass, cos0_lab)
    q = momentum_transfer_cm(omega_cm, cos0_cm)
    q = np.max([q, omega_cm], axis=0)
    return np.squeeze((q + mass_pion) / breakdown)


def coefficients(y, ratio, ref=1, orders=None):
    """Returns the coefficients of a power series

    Parameters
    ----------
    y : array, shape = (n_samples, n_curves)
    ratio : scalar or array, shape = (n_samples,)
    ref : scalar or array, shape = (n_samples,)
    orders : 1d array, optional
        The orders at which y was computed. Defaults to 0, 1, ..., n_curves-1

    Returns
    -------
    An (n_samples, n_curves) array of the extracted coefficients
    """
    if y.ndim != 2:
        raise ValueError('y must be 2d')
    if orders is None:
        orders = np.arange(y.shape[-1])
    if orders.shape[-1] != y.shape[-1]:
        raise ValueError('partials and orders must have the same length')

    ref, ratio, orders = np.atleast_1d(ref, ratio, orders)
    ref = ref[:, None]
    ratio = ratio[:, None]

    # Make coefficients
    coeffs = np.diff(y, axis=-1)                       # Find differences
    coeffs = np.insert(coeffs, 0, y[..., 0], axis=-1)  # But keep leading term
    coeffs = coeffs / (ref * ratio**orders)            # Scale each order appropriately
    return coeffs


def compute_idx_mat(n):
    idx = np.arange(n)
    idx_rows, idx_cols = np.broadcast_arrays(idx[:, None], idx)
    idx_mat = np.dstack([idx_rows, idx_cols])
    return idx_mat


def p_sq_grad_coeff_mat(n):
    n_rows = int(n * (n + 1) / 2)
    idx_mat = compute_idx_mat(n)
    idx_vec = idx_mat[np.triu_indices(n)]

    p_sq_grad = np.zeros((n_rows, n))
    for i in range(n):
        p_sq_grad[:, i] = np.sum(idx_vec == i, axis=1)
    return p_sq_grad


def p_sq_grad_idx_mat(n):
    idx_mat = compute_idx_mat(n)
    idx1, idx2 = np.triu_indices(idx_mat.shape[0])
    idx_mat_tri = idx_mat[idx1, idx2, :]
    n_rows = int(n * (n + 1) / 2)
    idx_mat = np.zeros((n_rows, n), dtype=int)
    for i in range(n):
        mask = np.any(idx_mat_tri == i, axis=1)
        idx_mat[mask, i] = np.arange(np.sum(mask), dtype=int)
    return idx_mat


def quadratic(x, A, b, c, flat=True):
    R"""Computes a multivariate quadratic function.

    Parameters
    ----------
    x : array, shape = (p,)
        The input variables
    A : array, shape = (N, p(p+1)/2,)
        The flattened quadratic coefficients
    b : array, shape = (N, p)
        The linear coefficients
    c : array, shape = (N,)
        The constant term
    flat

    Returns
    -------
    array, shape = (N,)
    """
    if flat:
        x = np.atleast_1d(x)
        x_sq = x[:, None] * x
        x_quad = x_sq[np.triu_indices_from(x_sq)]
        quad = A @ x_quad
    else:
        quad = np.einsum('...ij,i,j', A, x, x)
    return quad + b @ x + c


def grad_quadratic(x, A, b, c, flat=True):
    R"""Computes the gradient of a multivariate quadratic function.

    Parameters
    ----------
    x : array, shape = (p,)
        The input variables
    A : array, shape = (N, p(p+1)/2)
        The flattened quadratic coefficients
    b : array, shape = (N, p)
        The linear coefficients
    c : array, shape = (N,)
        The constant term
    flat

    Returns
    -------
    array, shape = (p, N)
    """
    if flat:
        x = np.atleast_1d(x)
        n = len(x)
        coeff_mat = p_sq_grad_coeff_mat(n)
        idx_mat = p_sq_grad_idx_mat(n)
        x_sq_grad = coeff_mat * x[idx_mat]
        quad = A @ x_sq_grad
    else:
        A_trans = np.swapaxes(A, -1, -2)
        quad = (A + A_trans) @ x
    return (quad + b).T


def quad_ratio(x, An, bn, cn, Ad, bd, cd, flat=True):
    R"""Computes the ratio of multivariate quadratic functions.

    Parameters
    ----------
    x : array, shape = (p,)
        The input variables
    An : array, shape = (N, p, p)
        The quadratic coefficients of the numerator
    bn : array, shape = (N, p)
        The linear coefficients of the numerator
    cn : array, shape = (N,)
        The constant term of the numerator
    Ad : array, shape = (N, p, p)
        The quadratic coefficients of the denominator
    bd : array, shape = (N, p)
        The linear coefficients of the denominator
    cd : array, shape = (N,)
        The constant term of the denominator
    flat

    Returns
    -------
    array, shape = (N,)
    """
    return quadratic(x, An, bn, cn, flat=flat) / quadratic(x, Ad, bd, cd, flat=flat)


def grad_quad_ratio(x, An, bn, cn, Ad, bd, cd, flat=True):
    R"""Computes the gradient of the ratio of multivariate quadratic functions.

    Parameters
    ----------
    x : array, shape = (p,)
        The input variables
    An : array, shape = (N, p, p)
        The quadratic coefficients of the numerator
    bn : array, shape = (N, p)
        The linear coefficients of the numerator
    cn : array, shape = (N,)
        The constant term of the numerator
    Ad : array, shape = (N, p, p)
        The quadratic coefficients of the denominator
    bd : array, shape = (N, p)
        The linear coefficients of the denominator
    cd : array, shape = (N,)
        The constant term of the denominator
    flat

    Returns
    -------
    array, shape = (p, N)
    """
    fn = quadratic(x, An, bn, cn, flat=flat)
    grad_fn = grad_quadratic(x, An, bn, cn, flat=flat)
    fd = quadratic(x, Ad, bd, cd, flat=flat)
    grad_fd = grad_quadratic(x, Ad, bd, cd, flat=flat)
    return grad_fn / fd - fn / fd ** 2 * grad_fd


def create_linearized_matrices(x0, An, bn, cn, Ad, bd, cd, flat=True):
    f0 = quad_ratio(x0, An, bn, cn, Ad, bd, cd, flat=flat)
    grad_f0 = grad_quad_ratio(x0, An, bn, cn, Ad, bd, cd, flat=flat)
    return f0 - x0 @ grad_f0, grad_f0.T


def shannon_expected_utility(X, cov_data, prec_p):
    R"""Computes the expected utility using the Shannon information, or the KL divergence

    X : np.ndarray, shape = (n_data, n_features)
        The feature matrix
    cov_data : np.ndarray, shape = (n_data, n_data)
        The covariance matrix for the data
    prec_p : np.ndarray, shape = (n_features, n_features)
        The prior precision on the parameters
    """
    _, log_det = slogdet(prec_p + X.T @ solve(cov_data, X))  # The negative of log |V|
    _, log_det_prec = slogdet(prec_p)  # The negative of log |V_0|
    return 0.5 * (log_det - log_det_prec)
    # p = prec_p.shape[0]
    # return 0.5 * (- p * log(2 * pi) - p + log_det)


def create_observable_set(df, cov_exp, p0_proton=None, cov_p_proton=None, p0_neutron=None,
                          cov_p_neutron=None, scale_dsg=True, p_transform=None):
    from compton import proton_pol_vec_mean, neutron_pol_vec_mean, proton_pol_vec_std, neutron_pol_vec_std
    proton_pol_cov = np.diag(proton_pol_vec_std)
    neutron_pol_cov = np.diag(neutron_pol_vec_std)

    groups = df.groupby(['observable', 'nucleon', 'order'])
    compton_obs = {}
    lin_vec = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    quad_vec = [col for col in df.columns if col[0] == 'C']
    for (obs, nucleon, order), index in groups.groups.items():
        if obs == 'crosssection':
            obs = 'dsg'

        df_i = df.loc[index]
        df_n = df_i[df_i['is_numerator'] == 1]
        df_d = df_i[df_i['is_numerator'] == 0]

        cov_p = None
        p0 = None
        if nucleon == 'proton':
            cov_p = cov_p_proton
            if cov_p_proton is None:
                cov_p = proton_pol_cov
            p0 = p0_proton
            if p0_proton is None:
                p0 = proton_pol_vec_mean
        elif nucleon == 'neutron':
            cov_p = cov_p_neutron
            if cov_p_neutron is None:
                cov_p = neutron_pol_cov
            p0 = p0_neutron
            if p0_neutron is None:
                p0 = neutron_pol_vec_mean

        obs_kwargs = dict(
            omega_lab=df_n['omegalab [MeV]'].values,
            degrees_lab=df_n['thetalab [deg]'].values,
            quad_n=df_n[quad_vec].values,
            lin_n=df_n[lin_vec].values,
            const_n=df_n['A'].values,
            quad_d=df_d[quad_vec].values,
            lin_d=df_d[lin_vec].values,
            const_d=df_d['A'].values,
            # name=obs if obs != 'crosssection' else 'dsg',
            name=obs,
            order=order,
            nucleon=nucleon,
            cov_p=cov_p,
            trans_mat=p_transform,
        )
        compton_obs[obs, nucleon, order, 'nonlinear'] = ComptonObservable(**obs_kwargs)

        try:
            cov_exp_i = cov_exp[obs, nucleon]
        except (TypeError, IndexError):
            if np.atleast_1d(cov_exp).ndim == 1:
                cov_exp_i = cov_exp * np.eye(len(df_n))
            else:
                cov_exp_i = cov_exp.copy()

        # if obs == 'crosssection' and scale_dsg:
        if (obs == 'dsg' or obs == r'$\sigma$') and scale_dsg:
            pred_i = compton_obs[obs, nucleon, order, 'nonlinear'](p0)
            cov_exp_i *= pred_i[:, None] * pred_i
        compton_obs[obs, nucleon, order, 'linear'] = ComptonObservable(**obs_kwargs, p0=p0, cov_data=cov_exp_i)
    return compton_obs


class ComptonObservable:
    R"""

    """

    def __init__(self, omega_lab, degrees_lab, quad_n, lin_n, const_n, quad_d, lin_d, const_d, order, name, nucleon,
                 p0=None, cov_data=None, cov_p=None, trans_mat=None):
        self.omega_lab = omega_lab
        self.degrees_lab = degrees_lab
        self.order = order
        self.name = name
        self.nucleon = nucleon
        self.linearized = False

        self.quad_n = quad_n
        self.lin_n = lin_n
        self.const_n = const_n
        self.quad_d = quad_d
        self.lin_d = lin_d
        self.const_d = const_d
        self.p0 = p0
        self.cov_data = cov_data
        self.cov_p = cov_p
        if cov_p is not None:
            self.prec_p = np.linalg.inv(cov_p)
        else:
            self.prec_p = None

        self.n_data = len(self.quad_n)

        if p0 is not None:
            self.linearized = True
            const, lin = create_linearized_matrices(p0, quad_n, lin_n, const_n, quad_d, lin_d, const_d, flat=True)
            self.const_approx = const
            self.lin_approx = lin
            if trans_mat is not None:
                self.lin_approx = lin @ np.linalg.inv(trans_mat)
            self.pred = self.prediction_linear
        else:
            self.const_approx = None
            self.lin_approx = None
            self.pred = self.prediction_ratio

    def __call__(self, p):
        return self.pred(p)

    def prediction_ratio(self, p):
        p_sq = p[:, None] * p
        p_quad = p_sq[np.triu_indices_from(p_sq)]
        num = self.quad_n @ p_quad + self.lin_n @ p + self.const_n
        den = self.quad_d @ p_quad + self.lin_d @ p + self.const_d
        return num / den

    def prediction_linear(self, p):
        return self.lin_approx @ p + self.const_approx

    def utility_linear(self, idx, p_idx=None):
        R"""Computes the expected shannon utility under the linear model assumption

        Parameters
        ----------
        idx : int or array
            The data set index used to denote the design of the experiment
        p_idx : int or array, optional
            The subset of theory parameters used in the computation of the expected utility. Defaults to `None`,
            which uses all theory parameters in the utility

        Returns
        -------
        expected_utility : float
        """
        X = self.lin_approx[idx]
        cov = self.cov_data[idx][:, idx]
        p_precision = self.prec_p
        if p_idx is not None:
            X = X[:, p_idx]
            p_precision = p_precision[p_idx][:, p_idx]
        return shannon_expected_utility(X, cov, p_precision)

    def __repr__(self):
        name = f'{self.name}({self.order}, {self.nucleon})'
        if self.p0 is not None:
            name += f' about {self.p0}'
        return name


class RBFJump(RBF):
    R"""An RBF Kernel that creates draws with a jump discontinuity in the function and all of its derivatives.

    See Scikit learn documentation for info on the original RBF kernel.
    The interesting new parameter is jump, which must have the same dimension as length_scale.
    This is the location of the jump, and the space with X < jump will be separated from X > jump.
    Thus, if dimension i has no jump, then one must set `jump[i] = np.inf`.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), jump=1.0):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.jump = jump

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError('gradients not implemented for jump kernel yet')
        K = super().__call__(X, Y=Y, eval_gradient=eval_gradient)
        if Y is None:
            Y = X
        mask_X = np.any(X > self.jump, axis=1)
        mask_Y = np.any(Y > self.jump, axis=1)
        # We want to find all pairs (x, x') where one is > jump and the other is < jump.
        # These points should be uncorrelated with one another.
        # We can use the XOR (exclusive or) operator to find all such pairs.
        zeros_mask = mask_X[:, None] ^ mask_Y
        K[zeros_mask] = 0.
        return K
