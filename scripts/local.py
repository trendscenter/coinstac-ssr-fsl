#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import sys
import simplejson as json
from ancillary import list_recursive, get_unique_phase_key
from local_funcs import local_1, local_2


def main():
    parsed_args = json.loads(sys.stdin.read())
    phase_keys = list_recursive(parsed_args, 'computation_phase')
    unique_phase_key = get_unique_phase_key(phase_keys)
    
    if not unique_phase_key:
        computation_output = local_1(parsed_args)
    elif "remote_1" in unique_phase_key:
        computation_output = local_2(parsed_args)
    else:
        raise ValueError("Error occurred at Local")

    sys.stdout.write(computation_output)


if __name__ == '__main__':
    main()
