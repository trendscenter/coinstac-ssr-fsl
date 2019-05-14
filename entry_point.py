#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:25:31 2019

@author: Harshvardhan
"""
import sys
import ujson as json
from function_chain import command_chain
from regression import list_recursive


def run_computation(function, arguments):
    try:
        computation_output = function(arguments)
        sys.stdout.write(computation_output)
    except Exception:
        raise ValueError("Error occurred at Remote")


def run_function(loc, key, args):
    try:
        func = command_chain.get(loc).get(key)
        if func is not None:
            run_computation(func, args)
        else:
            raise Exception('No such preceding function exists')
    except AttributeError:
        raise Exception('Location has to be either local or remote')


def get_phase_key(parsed_args):
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    unique_phase_key = list(set(phase_key))

    if len(unique_phase_key) > 1:
        raise Exception('Phase Key is not unique')
    elif not len(unique_phase_key):
        phase_key = 'start'
    else:
        phase_key = unique_phase_key[0]

    return phase_key


def read_input():
    return json.loads(sys.stdin.read())


def main(location):
    parsed_args = read_input()
    phase_key = get_phase_key(parsed_args)
    run_function(location, phase_key, parsed_args)
