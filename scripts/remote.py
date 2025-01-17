#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import sys
import simplejson as json
from ancillary import list_recursive, get_unique_phase_key
from remote_funcs import remote_1 ,remote_2


def main():
    parsed_args = json.loads(sys.stdin.read())
    phase_keys = list_recursive(parsed_args, 'computation_phase')
    unique_phase_key = get_unique_phase_key(phase_keys)

    if "local_1" in unique_phase_key:
        computation_output = remote_1(parsed_args)
    elif "local_2" in unique_phase_key:
        computation_output = remote_2(parsed_args)
    else:
        raise ValueError("Error occurred at Remote")
        
    sys.stdout.write(computation_output)


if __name__ == '__main__':
    main()
