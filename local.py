#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import regression as reg
import statsmodels.api as sm
import sys
import ujson as json
from parsers import fsl_parser
from local_ancillary import local_stats_to_dict_fsl, ignore_nans


def local_1(args):
    """Computes local beta vector

    Args:
        args (dictionary) : {"input": {
                                "covariates": ,
                                 "data": ,
                                 "lambda": ,
                                },
                            "cache": {}
                            }

    Returns:
        computation_output(json) : {"output": {
                                        "beta_vector_local": ,
                                        "mean_y_local": ,
                                        "count_local": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "covariates": ,
                                        "dependents": ,
                                        "lambda":
                                        }
                                    }

    Comments:
        Step 1 : Generate the local beta_vector
        Step 2 : Compute mean_y_local and length of target values

    """
    input_list = args["input"]
    (X, y) = fsl_parser(args)

    X_labels = ['const'] + list(X.columns)
    y_labels = list(y.columns)

    lamb = input_list["lambda"]

    t = local_stats_to_dict_fsl(X, y)
    beta_vector, local_stats_list, meanY_vector, lenY_vector = t

    output_dict = {
        "beta_vector_local": beta_vector,
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "X_labels": X_labels,
        "y_labels": y_labels,
        "local_stats_dict": local_stats_list,
        "computation_phase": 'local_1',
    }

    cache_dict = {
        "covariates": X.to_json(orient='split'),
        "dependents": y.to_json(orient='split'),
        "lambda": lamb
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
    
    return json.dumps(computation_output)


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = pd.read_json(cache_list["covariates"], orient='split')
    y = pd.read_json(cache_list["dependents"], orient='split')
    biased_X = sm.add_constant(X.values)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local, SST_local, varX_matrix_local = [], [], []
    for index, column in enumerate(y.columns):
        curr_y = y[column]

        X_, y_ = ignore_nans(biased_X, curr_y)

        SSE_local.append(reg.sum_squared_error(X_, y_, avg_beta_vector[index]))
        SST_local.append(
            np.sum(np.square(np.subtract(y_, mean_y_global[index]))))

        varX_matrix_local.append(np.dot(X_.T, X_).tolist())

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local,
        "computation_phase": 'local_2'
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
