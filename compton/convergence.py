import numpy as np
from numpy.linalg import slogdet, solve
from numpy import log, pi
import pandas as pd
from scipy.special import expit
from .constants import mass_pion
from .kinematics import momentum_transfer_cm, cos0_cm_from_lab, omega_cm_from_lab
from .constants import omega_lab_cusp, dsg_label, DesignLabels
from sklearn.gaussian_process.kernels import RBF
import gsum as gm


def order_transition(n, n_inf, omega):
    return n + (n_inf - n) * expit((omega-190)/20)


def expansion_parameter(X, breakdown):
    X = np.atleast_2d(X)
    return np.squeeze((X[:, 0] + mass_pion) / breakdown)


def expansion_parameter_phillips(breakdown, factor=1):
    return np.sqrt(mass_pion * factor / breakdown)


def expansion_parameter_cm(X, breakdown, mass, factor=1.):
    X = np.atleast_2d(X)
    omega_lab, _ = X.T
    omega_cm = omega_cm_from_lab(omega_lab, mass=mass)
    num = omega_cm + mass_pion
    num = num * factor
    # num = (omega_cm + mass_pion) / 2
    return np.squeeze(np.sqrt(num / breakdown))


def expansion_parameter_momentum_transfer_cm(X, breakdown, mass, include_correction=False):
    X = np.atleast_2d(X)
    omega_lab, cos0_lab = X.T
    cos0_lab = np.cos(np.deg2rad(cos0_lab))
    omega_cm = omega_cm_from_lab(omega_lab, mass=mass)
    cos0_cm = cos0_cm_from_lab(omega_lab, mass, cos0_lab)
    # q = momentum_transfer_cm(omega_cm, cos0_cm)
    num = omega_cm + mass_pion
    # num = (omega_cm + mass_pion) / 2

    if include_correction:
        # height = 200
        # omega_width = 50
        height = 150
        omega_width = 150
        cos0_width = 1
        lorentz = height / (
                ((omega_lab - omega_lab_cusp) / omega_width) ** 2 + ((cos0_lab - 1) / cos0_width) ** 2 + 1
        )
        num += lorentz
    from scipy.special import softmax, logsumexp
    # num = softmax([q, omega_cm], axis=0)
    # num = logsumexp([q, omega_cm], axis=0)
    # num = (q + omega_cm) / 2.
    # return np.squeeze(num / breakdown)
    return np.squeeze(np.sqrt(num / breakdown))


def compute_expansion_summation_matrix(Q, first_omitted_order):
    Q_mat = Q[:, None] * Q
    Q_to_omitted = Q ** first_omitted_order
    return Q_to_omitted[:, None] * Q_to_omitted / (1 - Q_mat)


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


def posterior_precision_linear(X, cov_data, prec_p):
    R"""Computes the posterior precision for parameters under a linear Gaussian model

        X : np.ndarray, shape = (n_data, n_features)
            The feature matrix
        cov_data : np.ndarray, shape = (n_data, n_data)
            The covariance matrix for the data
        prec_p : np.ndarray, shape = (n_features, n_features)
            The prior precision on the parameters
        """
    return prec_p + X.T @ solve(cov_data, X)


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


def create_observable_set(df, cov_exp=0., p0_proton=None, cov_p_proton=None, p0_neutron=None,
                          cov_p_neutron=None, scale_dsg=True, p_transform=None, expts_info=None):
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
        if nucleon == DesignLabels.proton:
            cov_p = cov_p_proton
            if cov_p_proton is None:
                cov_p = proton_pol_cov
            p0 = p0_proton
            if p0_proton is None:
                p0 = proton_pol_vec_mean
        elif nucleon == DesignLabels.neutron:
            cov_p = cov_p_neutron
            if cov_p_neutron is None:
                cov_p = neutron_pol_cov
            p0 = p0_neutron
            if p0_neutron is None:
                p0 = neutron_pol_vec_mean
        else:
            raise ValueError('nucleon must be Proton or Neutron')

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

        if expts_info is None:
            try:
                cov_exp_i = cov_exp[obs, nucleon]
            except (TypeError, IndexError):
                if np.atleast_1d(cov_exp).ndim == 1:
                    cov_exp_i = cov_exp * np.eye(len(df_n))
                else:
                    cov_exp_i = cov_exp.copy()

            # if obs == 'crosssection' and scale_dsg:
            if (obs == 'dsg' or obs == dsg_label) and scale_dsg:
                pred_i = compton_obs[obs, nucleon, order, 'nonlinear'](p0)
                cov_exp_i *= pred_i[:, None] * pred_i
        else:
            if nucleon == DesignLabels.proton:
                cov_exp_i = expts_info[obs].cov_proton
            elif nucleon == DesignLabels.neutron:
                cov_exp_i = expts_info[obs].cov_neutron
            else:
                raise ValueError('nucleon must be Proton or Neutron')
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
        self.trans_mat = trans_mat

        if (cov_p is not None) and not np.allclose(cov_p, cov_p.T):
            print(f'Warning: Parameter covariance is not symmetric. name={name}, nucleon={nucleon}')
        if (cov_data is not None) and not np.allclose(cov_data, cov_data.T):
            print(f'Warning: Data covariance is not symmetric. name={name}, nucleon={nucleon}')

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
        if self.trans_mat is not None:
            p = np.linalg.inv(self.trans_mat) @ p
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

    def correlation_matrix(self, idx, p_idx=None):
        X = self.lin_approx[idx]
        cov = self.cov_data[idx][:, idx]
        p_precision = self.prec_p
        if p_idx is not None:
            X = X[:, p_idx]
            p_precision = p_precision[p_idx][:, p_idx]
        print(np.count_nonzero(cov - cov.T))
        post_cov = np.linalg.inv(posterior_precision_linear(X, cov, p_precision))
        post_stds = np.sqrt(np.diag(post_cov))
        print(post_cov)
        return post_stds[:, None]**(-1) * post_cov * post_stds**(-1)

    def __repr__(self):
        name = f'{self.name}({self.order}, {self.nucleon})'
        if self.p0 is not None:
            name += f' about {self.p0}'
        return name


class ConvergenceAnalyzer:
    exp_param_funcs = {
        'sum': expansion_parameter_cm,
        # 'halfsum': lambda *args, **kwargs: expansion_parameter_cm(*args, **kwargs, factor=0.5),
        'halfsum': expansion_parameter_cm,
        'sumsq': lambda *args, **kwargs: expansion_parameter_cm(*args, **kwargs) ** 2,
        'phillips': expansion_parameter_phillips,
    }

    def __init__(
            self, name, nucleon, X, y, orders, train, ref, breakdown, excluded, exp_param='sum',
            delta_transition=True, degrees_zeros=None, omega_zeros=None,
            degrees_deriv_zeros=None, omega_deriv_zeros=None, **kwargs
    ):
        from compton import DesignLabels, mass_proton, mass_neutron
        self.name = name
        self.X = X
        self.y = y
        self.orders = orders
        self.train = train
        self.excluded = excluded
        self.ref = ref
        self.breakdown = breakdown
        self.kwargs = kwargs

        self.degrees_zeros = degrees_zeros
        self.omega_zeros = omega_zeros
        self.degrees_deriv_zeros = degrees_deriv_zeros
        self.omega_deriv_zeros = omega_deriv_zeros

        # from gsum import cartesian
        # self.X_zeros = cartesian(self.omega_zeros, self.degrees_zeros)

        self.exp_param = exp_param
        self.exp_param_func = self.exp_param_funcs[exp_param]

        if nucleon == DesignLabels.proton:
            mass = mass_proton
        elif nucleon == DesignLabels.neutron:
            mass = mass_neutron
        else:
            raise ValueError('nucleon must be DesignLabels.proton or DesignLabels.neutron')

        self.nucleon = nucleon
        self.mass = mass

        included = ~ np.isin(orders, excluded)
        self.included = included

        if delta_transition:
            from compton.constants import order_map
            # order_map = {0: 0, 2: 1, 3: 2, 4: 3}
            ord_vals = np.array([order_transition(order, order_map[order], X[:, 0]) for order in orders]).T
        else:
            ord_vals = np.array([np.broadcast_to(order, X.shape[0]) for order in orders]).T
        self.ord_vals = ord_vals

        if exp_param == 'sum':
            Q = expansion_parameter_cm(X, breakdown, mass=mass)
        elif exp_param == 'halfsum':
            Q = expansion_parameter_cm(X, breakdown, mass=mass, factor=0.5)
        elif exp_param == 'sumsq':
            Q = expansion_parameter_cm(X, breakdown, mass=mass)**2
        elif exp_param == 'phillips':
            Q = np.broadcast_to(expansion_parameter_phillips(breakdown), X.shape[0])
        else:
            raise ValueError('')
        self.Q = Q

        self.c = c = coefficients(y, Q, ref, ord_vals)
        self.c_included = c[:, included]

        self.X_train = self.X[train]
        self.y_train = self.y[train][:, included]
        self.c_train = c[train][:, included]

        gp = gm.ConjugateGaussianProcess(**kwargs)
        print(self.c_train.shape)
        gp.fit(self.X_train, self.c_train)
        print('Fit kernel:', gp.kernel_)
        self.cbar = np.sqrt(gp.cbar_sq_mean_)
        print('cbar mean:', self.cbar)
        self.gp = gp

    def compute_conditional_cov(self, X, gp=None):
        if gp is None:
            gp = gm.ConjugateGaussianProcess(**self.kwargs)
            gp.fit(self.X_train, self.c_train)

        if self.degrees_zeros is None and self.omega_zeros is None:
            return gp.cov(X)

        [ls_omega, ls_degrees] = gp.kernel_.k1.get_params()['length_scale']
        std = np.sqrt(gp.cbar_sq_mean_)

        w = X[:, [0]]
        t = X[:, [1]]

        import gptools

        kern_omega = gptools.SquaredExponentialKernel(
            initial_params=[1, ls_omega], fixed_params=[True, True])
        kern_theta = gptools.SquaredExponentialKernel(
            initial_params=[1, ls_degrees], fixed_params=[True, True])
        gp_omega = gptools.GaussianProcess(kern_omega)
        gp_theta = gptools.GaussianProcess(kern_theta)
        # gp_omega.add_data(np.array([[0], [0]]), np.array([0, 0]), n=np.array([0, 1]))

        if self.omega_zeros is not None or self.omega_deriv_zeros is not None:
            w_z = []
            n_w = []

            if self.omega_zeros is not None:
                w_z.append(self.omega_zeros)
                n_w.append(np.zeros(len(self.omega_zeros)))
            if self.omega_deriv_zeros is not None:
                w_z.append(self.omega_deriv_zeros)
                n_w.append(np.ones(len(self.omega_deriv_zeros)))
            w_z = np.concatenate(w_z)[:, None]
            n_w = np.concatenate(n_w)
            print(w_z, n_w)

            gp_omega.add_data(w_z, np.zeros(w_z.shape[0]), n=n_w)
            _, K_omega = gp_omega.predict(w, np.zeros(w.shape[0]), return_cov=True)
        else:
            K_omega = gp_omega.compute_Kij(w, w, np.zeros(w.shape[0]), np.zeros(w.shape[0]))

        if self.degrees_zeros is not None or self.degrees_deriv_zeros is not None:
            t_z = []
            n_t = []

            if self.degrees_zeros is not None:
                t_z.append(self.degrees_zeros)
                n_t.append(np.zeros(len(self.degrees_zeros)))
            if self.degrees_deriv_zeros is not None:
                t_z.append(self.degrees_deriv_zeros)
                n_t.append(np.ones(len(self.degrees_deriv_zeros)))
            t_z = np.concatenate(t_z)[:, None]
            n_t = np.concatenate(n_t)

            gp_theta.add_data(t_z, np.zeros(t_z.shape[0]), n=n_t)
            _, K_theta = gp_theta.predict(t, np.zeros(t.shape[0]), return_cov=True)
        else:
            K_theta = gp_theta.compute_Kij(t, t, np.zeros(t.shape[0]), np.zeros(t.shape[0]))

        # kernel_omega = RBF(ls_omega)
        # kernel_theta = RBF(ls_degrees)

        # if self.omega_zeros is not None:
        #
        #     w_z = np.atleast_1d(self.omega_zeros)[:, None]
        #
        #     K_omega = kernel_omega(w) - kernel_omega(w, w_z) @ np.linalg.solve(kernel_omega(w_z), kernel_omega(w_z, w))
        # else:
        #     K_omega = kernel_omega(w)
        #
        # if self.degrees_zeros is not None:
        #     t_z = np.atleast_1d(self.degrees_zeros)[:, None]
        #     K_theta = kernel_theta(t) - kernel_theta(t, t_z) @ np.linalg.solve(kernel_theta(t_z), kernel_theta(t_z, t))
        # else:
        #     K_theta = kernel_theta(t)

        return std**2 * K_omega * K_theta

        # X_z = self.X_zeros
        # K_nn = gp.cov(X)
        # K_zz = gp.cov(X_z)
        # K_nz = gp.cov(X, X_z)
        # K_zn = gp.cov(X_z, X)
        # return K_nn - K_nz @ np.linalg.solve(K_zz, K_zn)

    def plot_coefficient_slices(self, omegas, thetas, axes=None):
        import matplotlib.pyplot as plt
        assert len(omegas) == len(thetas)
        n = len(omegas)
        if axes is None:
            fig, axes = plt.subplots(n, 2, figsize=(3.4, 1.2 * n), sharex='col', sharey=True)
        fig = plt.gcf()

        ymax_w = 0
        ymax_t = 0

        color_list = ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples', 'Greys']
        cmaps = [plt.get_cmap(name) for name in color_list]
        # colors = [cmap(0.65 - 0.1 * (i == 0)) for i, cmap in enumerate(cmaps)]
        colors = ['k', plt.get_cmap('Greens')(0.8), plt.get_cmap('Blues')(0.65), plt.get_cmap('Reds')(0.6)]

        # cbar = self.cbar

        # linestyles = ['-', '--', '-.', ':']
        # linestyles = ['-', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
        linestyles = [(0, (5, 1)), '-', (0, (3, 1, 1, 1)), (0, (1, 1)), ]
        linewidths = [1, 1.0, 1.1, 1.1]

        cov = self.compute_conditional_cov(self.X)
        std = np.sqrt(np.diag(cov))
        # cov = self.gp.cov(self.X)
        for i, (omega_i, theta_i) in enumerate(zip(omegas, thetas)):
            ax_w, ax_t = axes[i]

            omega_mask = self.X[:, 1] == theta_i
            theta_mask = self.X[:, 0] == omega_i
            omega_vals = self.X[omega_mask, 0]
            theta_vals = self.X[theta_mask, 1]
            std_omega = np.sqrt(np.diag(cov[omega_mask][:, omega_mask]))
            std_theta = np.sqrt(np.diag(cov[theta_mask][:, theta_mask]))
            orders = self.orders
            c_w = self.c[omega_mask]
            c_t = self.c[theta_mask]

            for j, n in enumerate(orders):
                ax_w.plot(
                    omega_vals, c_w[:, j], color=colors[j], label=f'$c_{{{n}}}$',
                    ls=linestyles[j], lw=linewidths[j], zorder=j/10
                )
                ax_t.plot(
                    theta_vals, c_t[:, j], color=colors[j], label=f'$c_{{{n}}}$',
                    ls=linestyles[j], lw=linewidths[j], zorder=j/10
                )

            ax_w.axhline(0, 0, 1, c='k', lw=1, zorder=-1)
            ax_t.axhline(0, 0, 1, c='k', lw=1, zorder=-1)
            bbox = dict(boxstyle='round', facecolor='w')

            ax_w.text(
                0.93, 0.9, fr'$\theta = {theta_i}^\circ$', transform=ax_w.transAxes,
                bbox=bbox, ha='right', va='top',
            )
            ax_t.text(
                0.93, 0.9, fr'$\omega = {omega_i}\,$MeV', transform=ax_t.transAxes,
                bbox=bbox, ha='right', va='top',
            )

            # ax_w.axhline(-2 * cbar, 0, 1, c='lightgrey', lw=1, zorder=-1)
            # ax_w.axhline(+2 * cbar, 0, 1, c='lightgrey', lw=1, zorder=-1)
            # ax_t.axhline(-2 * cbar, 0, 1, c='lightgrey', lw=1, zorder=-1)
            # ax_t.axhline(+2 * cbar, 0, 1, c='lightgrey', lw=1, zorder=-1)
            std_lw = 1.2
            # ax_w.plot(omega_vals, + 2 * std_omega, c='lightgrey', lw=std_lw, zorder=-1, label=r'$2\sigma$')
            # ax_w.plot(omega_vals, - 2 * std_omega, c='lightgrey', lw=std_lw, zorder=-1)
            # ax_t.plot(theta_vals, + 2 * std_theta, c='lightgrey', lw=std_lw, zorder=-1, label=r'$2\sigma$')
            # ax_t.plot(theta_vals, - 2 * std_theta, c='lightgrey', lw=std_lw, zorder=-1)

            ax_w.fill_between(
                omega_vals, + 2 * std_omega, - 2 * std_omega, facecolor='0.92', lw=0.7,
                zorder=-1, label=r'$2\sigma$', edgecolor='0.6'
            )
            ax_t.fill_between(
                theta_vals, + 2 * std_theta, - 2 * std_theta, facecolor='0.92', lw=0.7,
                zorder=-1, label=r'$2\sigma$', edgecolor='0.6'
            )

            ymax_w = np.max(np.abs(ax_w.get_ylim()))
            ymax_t = np.max(np.abs(ax_t.get_ylim()))

        ymax = np.max([ymax_w, ymax_t])
        if ymax > 4.2 * np.max(std):
            ymax = 4.2 * np.max(std)

        if self.nucleon == 'Neutron':
            title = f'{self.name}, {self.nucleon}'
        else:
            title = f'{self.name}'
        # with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
        axes[0, 0].text(
            0.07, 0.9, title, transform=axes[0, 0].transAxes,
            bbox=bbox, ha='left', va='top',
        )
            # plt.draw()

        ax_w.set_ylim(-ymax, ymax)
        ax_t.set_ylim(-ymax, ymax)
        ax_w.set_xlabel(r'$\omega_{\mathrm{lab}}$\,[MeV]')
        ax_t.set_xlabel(r'$\theta_{\mathrm{lab}}$\,[deg]')
        ax_w.set_xticks([100, 200, 300])
        ax_w.set_xticks([50, 150, 250], minor=True)
        ax_t.set_xticks([60, 120])
        ax_t.set_xticks([30, 90, 150], minor=True)
        ax_w.set_xlim(self.X[:, 0].min(), 340)
        ax_t.set_xlim(self.X[:, 1].min(), self.X[:, 1].max())
        # fig.suptitle(f'{self.name}, {self.nucleon}')
        # axes[0, 0].set_title(f'{self.name}, {self.nucleon}')
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator
        axes[0, 0].yaxis.set_major_locator(MaxNLocator(3))
        axes[0, 0].yaxis.set_minor_locator(AutoMinorLocator(2))

        for ax in axes.ravel():
            ax.tick_params(right=True, top=True, which='both')

        fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=1 / 72)
        plt.draw()
        upper_right_display = axes[0, 1].transAxes.transform((1, 1))
        upper_right_axes00 = axes[0, 0].transAxes.inverted().transform(upper_right_display)
        axes[0, 0].legend(
            loc='lower left', bbox_to_anchor=(0, 1.03, upper_right_axes00[0], 0), borderaxespad=0, ncol=5,
            mode='expand',
            columnspacing=0,
            handletextpad=0.5,
            # handlelength=1.2,
            fancybox=False,
        )

        return axes

    def log_marginal_likelihood(self, **kwargs):

        included = self.included
        orders = self.orders
        ord_vals = self.ord_vals
        ref = self.ref
        train = self.train
        breakdown = self.breakdown
        X = self.X
        mass = self.mass

        if self.exp_param == 'sum' or self.exp_param == 'halfsum':
            Q = expansion_parameter_cm(X, breakdown, mass=mass, **kwargs)
        elif self.exp_param == 'sumsq':
            Q = expansion_parameter_cm(X, breakdown, mass=mass, **kwargs)**2
        elif self.exp_param == 'phillips':
            Q = expansion_parameter_phillips(breakdown, **kwargs)
            Q = np.broadcast_to(Q, X.shape[0])
        else:
            raise ValueError()
        coeffs = coefficients(self.y, Q, self.ref, ord_vals)
        coeffs = coeffs[self.train][:, included]

        gp = gm.ConjugateGaussianProcess(**self.kwargs)
        gp.fit(self.X_train, coeffs)

        # K = self.compute_conditional_cov(self.X_train, gp)
        # alpha = np.linalg.solve(K, coeffs)
        # coeff_log_like = -0.5 * np.einsum('ik,ik->k', coeffs, alpha) - \
        #     0.5 * np.linalg.slogdet(2 * np.pi * K)[-1]
        # coeff_log_like = coeff_log_like.sum()

        coeff_log_like = gp.log_marginal_likelihood_value_

        orders_in = orders[included]
        n = len(orders_in)

        try:
            ref_train = ref[train]
        except TypeError:
            ref_train = ref

        det_factor = np.sum(
            n * np.log(np.abs(ref_train)) +
            np.sum(ord_vals[train][:, included], axis=1) * np.log(np.abs(Q[train]))
        )
        y_log_like = coeff_log_like - det_factor
        return y_log_like


class RBFJump(RBF):
    R"""An RBF Kernel that creates draws with a jump discontinuity in the function and all of its derivatives.

    See Scikit learn documentation for info on the original RBF kernel.
    The interesting new parameter is jump, which must have the same dimension as length_scale.
    This is the location of the jump, and the space with X < jump will be separated from X > jump.
    Thus, if dimension i has no jump, then one must set `jump[i] = np.inf`.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), jump=None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.jump = jump

    def __call__(self, X, Y=None, eval_gradient=False):
        # if eval_gradient:
        #     raise ValueError('gradients not implemented for jump kernel yet')
        K = super().__call__(X, Y=Y, eval_gradient=eval_gradient)
        if self.jump is None:
            return K

        if Y is None:
            Y = X
        mask_X = np.any(X > self.jump, axis=1)
        mask_Y = np.any(Y > self.jump, axis=1)
        # We want to find all pairs (x, x') where one is > jump and the other is < jump.
        # These points should be uncorrelated with one another.
        # We can use the XOR (exclusive or) operator to find all such pairs.
        zeros_mask = mask_X[:, None] ^ mask_Y

        if eval_gradient:
            K, dK = K
            K[zeros_mask] = 0.
            dK[zeros_mask] = 0.
            return K, dK

        K[zeros_mask] = 0.
        return K


from sklearn.gaussian_process.kernels import Kernel


class ConditionalKernel(RBFJump):
    R"""
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), jump=None, X=None, dim=None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, jump=jump)
        # self.k = k
        self.X = X
        self.dim = dim

    def __call__(self, X, Y=None, eval_gradient=False):
        X_cond = self.X

        if self.dim is not None:
            X = X[:, [self.dim]]
            if Y is not None:
                Y = Y[:, [self.dim]]

        if self.X is None:
            return super().__call__(X, Y, eval_gradient)

        K_nn = super().__call__(X, Y, eval_gradient)
        K_oo = super().__call__(X_cond, eval_gradient=eval_gradient)
        K_no = super().__call__(X, X_cond, eval_gradient=False)

        if eval_gradient:
            K_nn, dK_nn = K_nn
            K_oo, dK_oo = K_oo
            # K_no, dK_no = K_no

        alpha = np.linalg.solve(K_oo, K_no.T)
        K = K_nn - K_no @ alpha

        if eval_gradient:
            # print(dK_oo.shape, K_no.shape)
            dK = dK_nn
            # d_alpha = np.linalg.solve(dK_oo.T, K_no)
            # dK = dK_nn - np.einsum('ij,kjl,ilk', K_no, d_alpha)
            # dK -= 2 * np.einsum('ijk,il->ilk', dK_no, alpha)
            return K, dK
        return K

    # def diag(self, X):
    #     return self.k.diag(X)
    #
    # def is_stationary(self):
    #     return self.k.is_stationary()
    #
    # def __repr__(self):
    #     return repr(self.k)

