#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:25:26 2018

@author: Harshvardhan
"""
import os
import pandas as pd
import nibabel as nib


def parse_for_y(args, X_files, y_files, y_labels):
    """Read contents of fsl files into a dataframe"""
    y = pd.DataFrame(index=y_labels)

    for file in y_files:
        if any([curr_X_file in file for curr_X_file in X_files]):
            y_ = pd.read_csv(
                os.path.join(args["state"]["baseDirectory"], file), sep='\t')
            y_.set_index('Measure:volume', inplace=True)
            y = pd.merge(y, y_, how='left', left_index=True, right_index=True)

    y = y.T

    return y


def fsl_parser(args):
    """Parse the freesurfer (fsl) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes"""
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]
    X_types = X_info[2]

    X_df = pd.DataFrame.from_records(X_data)

    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))
    X_files = list(X_df['freesurferfile'])

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = X * 1

    y_files = y_info[0]
    y_labels = y_info[2]

    y = parse_for_y(args, X_files, y_files, y_labels)

    X = X.reindex(sorted(X.columns), axis=1)

    return (X, y)


def nifti_to_data(args, X_files, y_files):
    """Read nifti files as matrices"""
    mask_file = os.path.join(args["state"]["baseDirectory"], 'mask_6mm.nii')
    mask_data = nib.load(mask_file).get_data()

    appended_data = []

    # Extract Data (after applying mask)
    for image in X_files:
        if image in y_files:
            image_data = nib.load(
                os.path.join(args["state"]["baseDirectory"],
                             image)).get_data()
            appended_data.append(image_data[mask_data > 0])

    return appended_data


def vbm_parser(args):
    """Parse the nifti (.nii) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes"""
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]
    #    X_types = X_info[2]

    X_df = pd.DataFrame.from_records(X_data)
    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))

    X_files = list(X_df['niftifile'])

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X = X * 1

    y_files = y_info[0]

    y_list = nifti_to_data(args, X_files, y_files)
    y = pd.DataFrame.from_records(y_list)

    y.columns = ['{}_{}'.format('voxel', str(i)) for i in y.columns]

    return (X, y)
