#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import numpy as np
import pandas as pd
import sys
import regression as reg
import warnings
from parsers import fsl_parser
from local_ancillary import gather_local_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


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

    y_labels = list(y.columns)

    lamb = input_list["lambda"]

    beta_vector, meanY_vector, lenY_vector, local_stats_list = gather_local_stats(
        args, X, y)

    output_dict = {
        "beta_vector_local": beta_vector,
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "y_labels": y_labels,
        "local_stats_dict": local_stats_list,
        "computation_phase": 'local_1',
    }

    cache_dict = {
        "covariates": X.to_json(orient='records'),
        "dependents": y.to_json(orient='records'),
        "lambda": lamb
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    raise Exception(y.to_json(orient='records'))
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

    X = pd.read_json(cache_list["covariates"], orient='records')
    y = pd.read_json(cache_list["dependents"], orient='records')
    biased_X = sm.add_constant(X.values)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = list(y[column])
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, avg_beta_vector))
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[index]))))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local.tolist(),
        "computation_phase": 'local_2'
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
