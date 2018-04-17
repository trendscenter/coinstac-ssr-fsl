""" Tests for local_ancillary.py """

import pytest
import numpy as np
import os
import sys

file_dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(file_dir_path)
sys.path.append(project_path)
import regression as reg



def test_one_shot_regression_normal():
    """ Tests that regression.one_shot_regression works for expected input """
    pass


def test_one_shot_regression_bad_input():
    """ Tests that regression.one_shot_regression fails for bad input """
    pass


def test_y_estimate_normal():
    """ Tests that regression.y_estimate works for expected input """
    biased_X = np.array([[1, 0], [2, 0], [3, 0]])
    beta_vector = [1, 0]
    expected = np.array([1, 2, 3])
    assert np.array_equal(reg.y_estimate(biased_X=biased_X, beta_vector=beta_vector), expected)


def test_y_estimate_bad_input():
    """ Tests that regression.y_estimate fails for bad input """
    pass


