""" Tests for regression.py 

I am mostly testing for one type of error below, namely, dimension mismatch between arrays. It would
be good to test corner cases as well. Examples could include negative numbers, NaNs, complex numbers, 
zeros, very large values, very small values, strings, or different data types such as lists or pandas 
DataFrames. I am not sure what types of corner-case inputs are likely to be encountered. Would be good 
to give this some thought or add tests that protect against common errors you have encountered. 

"""

import os
import sys
import pytest
import numpy as np
import numpy.testing as npt

file_dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(file_dir_path)
sys.path.append(project_path)

import regression as reg


def test_one_shot_regression_normal():
    """ Tests that regression.one_shot_regression works for expected input """
    X = np.array([1, 2, 3, 4])
    y = [1, 2, 3, 4]
    expected_beta_vector = np.array([1.])
    npt.assert_allclose(reg.one_shot_regression(X, y, 0), expected_beta_vector)


def test_one_shot_regression_reg_check():
    """ Tests that in regression.one_shot_regression the regularization parameter decreases beta weights """
    X = np.array([1, 2, 3, 4])
    y = [1, 2, 3, 4]
    lamb = 1000000
    expected_beta_vector = np.array([0.])
    npt.assert_allclose(reg.one_shot_regression(X, y, lamb), expected_beta_vector, atol=1e-3)


def test_one_shot_regression_dim_mismatch():
    """ Tests that regression.one_shot_regression raises ValueError when array dimensions do not match """
    X = np.array([1, 2, 3, 4])
    y = [1, 2, 3]
    with pytest.raises(ValueError):
        reg.one_shot_regression(X, y, 0)


def test_y_estimate_normal():
    """ Tests that regression.y_estimate works for expected input """
    biased_X = np.array([[1, 0], [2, 0], [3, 0]])
    beta_vector = [1, 0]
    expected = np.array([1, 2, 3])
    npt.assert_allclose(reg.y_estimate(biased_X=biased_X, beta_vector=beta_vector), expected)


def test_y_estimate_dim_mismatch():
    """ Tests that regression.y_estimate raises ValueError when array dimensions do not match """
    biased_X = np.array([[1, 0, 1], [2, 0, 1], [3, 0, 1]])
    beta_vector = [1, 0]
    with pytest.raises(ValueError):
        reg.y_estimate(biased_X=biased_X, beta_vector=beta_vector)


def test_y_estimate_list_input():
    """ Tests that regression.y_estimate works for expected input """
    # TODO: Is this the right behavior? Should we require that input be a Numpy array, or should we convert
    # TODO: to a Numpy array inside the function?

    biased_X = [[1, 0], [2, 0], [3, 0]]
    beta_vector = [1, 0]
    expected = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        reg.y_estimate(biased_X=biased_X, beta_vector=beta_vector)


def test_sum_squared_error_normal():
    """ Tests that regression.sum_squared_error works for expected input """
    biased_X = np.array([[1, 0], [2, 0], [3, 0]])
    beta_vector = [1, 0]
    y = np.array([3, 2, 3])  # y_estimate should be [1, 2, 3]
    expected = 4.0
    npt.assert_allclose(reg.sum_squared_error(biased_X, y, beta_vector), expected)


def test_sum_squared_error_dim_mismatch():
    """ Tests that regression.sum_squared_error raises ValueError when array dimensions do not match """
    biased_X = np.array([[1, 0, 1], [2, 0, 1], [3, 0, 1]])
    beta_vector = [1, 0]
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        reg.sum_squared_error(biased_X, y, beta_vector)


def test_sum_squared_total_normal():
    """ Tests that regression.sum_squared_total works for expected input """
    y = np.array([-1, 0, 1])
    expected = 2
    npt.assert_allclose(reg.sum_squared_total(y), expected)


def test_sum_squared_total_empty():
    """ Tests that regression.sum_squared_total works for empty input """
    # TODO: Do we want to raise an exception or have the output be zero, None, [], or something else?
    y = np.array([])
    expected = 0
    npt.assert_allclose(reg.sum_squared_total(y), expected)



