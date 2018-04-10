#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:25:26 2018

@author: Harshvardhan
"""
import os
import pandas as pd


def parse_for_each_y(args, X_files, y_files, dependent):
    y = []
    for file in y_files:
        if file.split('-')[-1] in X_files:
            with open(
                    os.path.join(args["state"]["baseDirectory"],
                                 file)) as fh:
                for line in fh:
                    if line.startswith(dependent[0]):
                        y.append(float(line.split('\t')[1]))

    return y


def parse_for_y_array(args, X_files, y_files, y_labels):
    y_array = []
    for region in y_labels:
        y_array.append(parse_for_each_y(args, X_files, y_files, region))

    y_array = list(map(list, zip(*y_array)))

    return y_array


def fsl_parser(args):
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]

    X_df = pd.DataFrame.from_records(X_data)
    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))

    X_files = list(X_df['freesurferfile'])

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')

    y_files = y_info[0]
    y_labels = y_info[2]

    y_list = parse_for_y_array(args, X_files, y_files, y_labels)
    y = pd.DataFrame.from_records(y_list, columns=y_labels)

    return (X, y, y_labels)  # Ask about labels being available at this level without being passed around
