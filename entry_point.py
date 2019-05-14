#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:25:31 2019

@author: Harshvardhan
"""
import sys
import ujson as json
from function_chain import command_chain


def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v


def run_function(function, arguments):
    try:
        computation_output = function(arguments)
        sys.stdout.write(computation_output)
    except Exception:
        raise ValueError("Error occurred at Remote")


def get_next_phase(loc, key):
    try:
        func = command_chain.get(loc).get(key)
        if func is None:
            raise Exception('No such preceding function exists')
        else:
            return func
    except AttributeError:
        raise Exception('Location has to be either local or remote')


def get_prev_phase(phase_key):
    unique_phase_key = list(set(phase_key))

    if len(unique_phase_key) > 1:
        raise Exception('Phase Key is not unique')
    elif not len(unique_phase_key):
        key = 'start'
    else:
        key = unique_phase_key[0]

    return key


def read_input():
    return json.loads(sys.stdin.read())


def main(location):
    parsed_args = read_input()
    key_list   = list_recursive(parsed_args, 'computation_phase')
    prev_func = get_prev_phase(key_list)
    next_func = get_next_phase(location, prev_func)
    run_function(next_func, parsed_args)
