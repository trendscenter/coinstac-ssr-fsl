#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import sys
import ujson as json
from ancillary import list_recursive
from local_funcs import local_1, local_2


def main():
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")


if __name__ == '__main__':
    main()
