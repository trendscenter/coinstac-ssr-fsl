#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import sys
import ujson as json
from scripts.ancillary import list_recursive, get_unique_phase_key
from scripts.remote_funcs import remote_1 ,remote_2


def start(parsed_args):
    phase_keys = list_recursive(parsed_args, 'computation_phase')
    unique_phase_key = get_unique_phase_key(phase_keys)

    if "local_1" in unique_phase_key:
        computation_output = remote_1(parsed_args)
    elif "local_2" in unique_phase_key:
        computation_output = remote_2(parsed_args)
    else:
        raise ValueError("Error occurred at Remote")
        
    return computation_output
