#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation

Example:
    python local.py '{"input":
                        {"covariates": [[2,3],[3,4],[7,8],[7,5],[9,8]],
                         "dependents": [6,7,8,5,6],
                         "lambda": 0
                         },
                     "cache": {}
                     }'
"""
import json
import numpy as np
import pandas as pd
import sys
import regression as reg
import warnings
from parsers import fsl_parser

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
    (X, y, y_labels) = fsl_parser(args)

    lamb = input_list["lambda"]
    biased_X = sm.add_constant(X)
    biased_X = biased_X.values

    beta_vector, meanY_vector, lenY_vector = [], [], []

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = list(y[column])
        beta = reg.one_shot_regression(biased_X, curr_y, lamb)
        beta_vector.append(beta.tolist())
        meanY_vector.append(np.mean(curr_y))
        lenY_vector.append(len(y))

        # Printing local stats as well
        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared_adj)

    keys = ["beta", "sse", "pval", "tval", "rsquared"]
    dict_list = []
    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
            local_rsquared[index]
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        dict_list.append(local_stats_dict)

    computation_output = {
        "output": {
            "beta_vector_local": beta_vector,
            "mean_y_local": meanY_vector,
            "count_local": lenY_vector,
            "y_labels": y_labels,
            "local_stats_dict": dict_list,
            "computation_phase": 'local_1',
        },
        "cache": {
            "covariates": X.values.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb
        }
    }

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

    X = cache_list["covariates"]
    y = cache_list["dependents"]
    biased_X = sm.add_constant(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    #    raise Exception(y, type(y))
    y = pd.DataFrame(y)

    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = list(y[column])
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, avg_beta_vector))
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[index]))))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": 'local_2'
        },
        "cache": {}
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
