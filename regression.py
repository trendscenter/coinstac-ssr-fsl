# -*- coding: utf-8 -*-
"""
This module contains functions to perform ridge regression and other relevant
functions including calculation of the coefficient of determination R^2 and
t-value
"""
import numpy as np
import scipy as sp
import warnings
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v


def one_shot_regression(X, y, lamb):
    """Performs ridge regression

    Args:
        X (float) : Training data of shape [n_samples, n_features]
        y (float) : Target values of shape [n_samples]
        lamb (float) : Regularization parameter lambda

    Returns:
        beta_vector (float) : weight vector of shape [n_features + 1]

    Comments:
        Utilizes sklearn.linear_model.Ridge to return a weight vector for the
        regression  model y = w*biased_X + epsilon
      """
#    clf = sklearn.linear_model.Ridge(
#        alpha=lamb,
#        fit_intercept=True,
#        normalize=False,
#        copy_X=True,
#        max_iter=None,
#        tol=0.001,
#        solver='auto',
#        random_state=None)
#
#    result = clf.fit(X, y)
#    beta_vector = np.insert(result.coef_, 0, result.intercept_)
    model = sm.OLS(y, X.astype(float)).fit_regularized(alpha=lamb, L1_wt=0)

    return model.params


def y_estimate(biased_X, beta_vector):
    """Returns the target estimates (predicted values of the target)

    Args:
        biased_X (float) : Augmented training data of shape
                            [n_samples, n_features + 1]
        beta_vector (float) : weight vector of shape [n_features + 1]

    Returns:
        numpy array of shape [n_samples]

    Comments:
        y_estimate = beta * biased_X'
    """
    return np.dot(beta_vector, np.matrix.transpose(biased_X))


def sum_squared_error(biased_X, y, beta_vector):
    """Calculates the sum of squared errors (SSE)

    Args:
        biased_X (float)    : Augmented training data of shape
                                [n_features + 1, n_samples]
            y               : Target values of shape [n_samples]
        beta_vector (float) : Weight vector of shape [n_features + 1]

    Returns:
        SSE (float)

    Comments:
        SSE = ||(y - y_estimate)^2||^2 where ||.|| --> l2-norm
    """
    return np.linalg.norm(y - y_estimate(biased_X, beta_vector))**2


def sum_squared_total(y):
    """Calculates the total sum of squares

    Args:
        y (float) : Target values of shape [n_samples]

    Returns:
        SST (float)

    Comments:
        SST = ||y - y_mean||^2 where ||.|| --> l2-norm
    """
    return np.linalg.norm(y - np.mean(y))**2


def r_square(biased_X, y, beta_vector):
    """Calculates R-squared value (coefficient of determination)

    Args:
        biased_X (float)    : Augmented traning data of shape
                                [n_features + 1, n_samples]
        y (float)           : Target values of shape [n_samples]
        beta_vector (float) : Weight vector of shape [n_features + 1]

    Returns:
        coefficient of determination (float)

    Comments:
        R^2 = 1 - SSE / SST
    """
    SSE = sum_squared_error(biased_X, y, beta_vector)
    SST = sum_squared_total(y)

    return 1 - SSE / SST


def beta_var_covar_matrix(biased_X, y, beta_vector):
    """Calculates the variance-covariance matrix of the coefficient vector

    Args:
        biased_X (float)    : Augmented training data of shape
                                [n_samples, n_features + 1]
        y (float)           : Target vector of shape [n_samples]
        beta_vector (float) : Coefficient array of shape [n_features + 1]

    Returns:
        var_covar_beta (float) : Square matrix of size [n_features + 1]

    Comments:
        Var(beta) = (sigma^2)*(X'*X)^(-1)
        Var(beta) = MSE*(X'*X)^(-1)
    """
    dof = len(y) - len(beta_vector)
    SSE = sum_squared_error(biased_X, y, beta_vector)
    MSE = SSE / dof

    return MSE * sp.linalg.inv(np.dot(biased_X.T, biased_X))


def t_value(biased_X, y, beta_vector):
    """Returns the t-statistic for each coefficient

    Args:
        biased_X (float)    : Augmented training data of shape
                                [n_samples, n_features + 1]
        y (float)           : Target vector of shape [n_samples]
        beta_vector (float) : Coefficient array of shape [n_features + 1]

    Returns:
        ts_beta (float) : t-values for each beta of shape [n_features + 1]

    Comments:
        t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
    """
    beta_variance = beta_var_covar_matrix(biased_X, y, beta_vector)
    se_beta = np.sqrt(beta_variance.diagonal())

    return beta_vector / se_beta


def t_to_p(ts_beta, dof):
    """Returns the p-value for each t-statistic of the coefficient vector

    Args:
        dof (int)       : Degrees of Freedom
                            Given by len(y) - len(beta_vector)
        ts_beta (float) : t-statistic of shape [n_features +  1]

    Returns:
        p_values (float): of shape [n_features + 1]

    Comments:
        t to p value transformation(two tail)
    """
    return [2 * stats.t.sf(np.abs(t), dof) for t in ts_beta]
