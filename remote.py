#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys
import scipy as sp
import numpy as np
import regression as reg


def remote_1(args):
    """Computes the global beta vector, mean_y_global & dof_global

    Args:
        args (dictionary): {"input": {
                                "beta_vector_local": list/array,
                                "mean_y_local": list/array,
                                "count_local": int,
                                "computation_phase": string
                                },
                            "cache": {}
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": list,
                                        "mean_y_global": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "avg_beta_vector": ,
                                        "mean_y_global": ,
                                        "dof_global":
                                        },
                                    }

    """
    input_list = args["input"]
    userId = list(input_list)[0]
    y_labels = input_list[userId]["y_labels"]  # don't like this line here because everyone has to sent the labels, but they should have been available at the remote itself by virtue of having specified in the compspec.json

    all_local_stats_dicts = [
        input_list[site]["local_stats_dict"] for site in input_list
    ]

    avg_beta_vector = np.average(
        [
            np.array(input_list[site]["beta_vector_local"])
            for site in input_list
        ],
        axis=0)

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [
        np.array(input_list[site]["count_local"]) for site in input_list
    ]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.average(mean_y_global, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    computation_output = {
        "output": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global.tolist(),
            "computation_phase": 'remote_1'
        },
        "cache": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global.tolist(),
            "dof_global": dof_global.tolist(),
            "y_labels": y_labels,
            "local_stats_dict": all_local_stats_dicts
        },
    }

    return json.dumps(computation_output)


def remote_2(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]
    y_labels = args["cache"]["y_labels"]
    all_local_stats_dicts = args["cache"]["local_stats_dict"]

    cache_list = args["cache"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum(
        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = avg_beta_vector[i] / se_beta_global
        ps = reg.t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    # Block of code to print local stats as well
    sites = ['Site_' + str(i) for i in range(len(all_local_stats_dicts))]

    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts)))

    a_dict = [{
        key: value
        for key, value in zip(sites, all_local_stats_dicts[i])
    } for i in range(len(all_local_stats_dicts))]

    # Block of code to print just global stats
    keys1 = [
        "avg_beta_vector", "r2_global", "ts_global", "ps_global", "dof_global"
    ]
    global_dict_list = []
    for index, _ in enumerate(y_labels):
        values = [
            avg_beta_vector[index], r_squared_global[index],
            ts_global[index].tolist(), ps_global[index], dof_global[index]
        ]
        my_dict = {key: value for key, value in zip(keys1, values)}
        global_dict_list.append(my_dict)

    # Print Everything
    dict_list = []
    keys2 = ["ROI", "global_stats", "local_stats"]
    for index, label in enumerate(y_labels):
        values = [label, global_dict_list[index], a_dict[index]]
        my_dict = {key: value for key, value in zip(keys2, values)}
        dict_list.append(my_dict)

    computation_output = {
        "output": {
            "regressions": dict_list
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_2" in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
