import numpy as np
from numpy.linalg import slogdet, solve
from numpy import log, pi
import pandas as pd


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
    p = prec_p.shape[-1]
    return 0.5 * (- p * log(2 * pi) - p + slogdet(prec_p + X.T @ solve(cov_data, X))[1])


class ComptonObservable:

    def __init__(self, omega_lab, degrees_lab, quad_n, lin_n, const_n, quad_d, lin_d, const_d, order, name, nucleon,
                 p0=None, cov_data=None, cov_p=None):
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

        if p0 is not None:
            self.linearized = True
            const, lin = create_linearized_matrices(p0, quad_n, lin_n, const_n, quad_d, lin_d, const_d, flat=True)
            self.const_approx = const
            self.lin_approx = lin
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

    def utility_linear(self, idx):
        X = self.lin_approx[idx]
        cov = self.cov_data[idx, idx.T]
        return shannon_expected_utility(X, cov, self.prec_p)

    def __repr__(self):
        name = f'{self.name}({self.order}, {self.nucleon})'
        if self.p0 is not None:
            name += f' about {self.p0}'
        return name
