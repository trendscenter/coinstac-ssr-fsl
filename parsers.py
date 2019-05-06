#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:25:26 2018

@author: Harshvardhan
"""
import nibabel as nib
import numpy as np
import os
import pandas as pd


def parse_for_y(args, y_files, y_labels):
    """Read contents of fsl files into a dataframe"""
    y = pd.DataFrame(index=y_labels)

    for file in y_files:
        if file:
            try:
                y_ = pd.read_csv(
                    os.path.join(args["state"]["baseDirectory"], file),
                    sep='\t',
                    header=None,
                    names=['Measure:volume', file],
                    index_col=0)
                y_ = y_[~y_.index.str.contains("Measure:volume")]
                y_ = y_.apply(pd.to_numeric, errors='ignore')
                y = pd.merge(
                    y, y_, how='left', left_index=True, right_index=True)
            except pd.errors.EmptyDataError:
                continue
            except FileNotFoundError:
                continue

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

    X_df = pd.DataFrame.from_records(X_data)

    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))
    X_df.set_index(X_df.columns[0], inplace=True)

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X = X * 1

    y_files = y_info[0]
    y_labels = y_info[2]

    y = parse_for_y(args, y_files, y_labels)

    X = X.reindex(sorted(X.columns), axis=1)

    ixs = X.index.intersection(y.index)

    if ixs.empty:
        raise Exception('No common X and y files at ' +
                        args["state"]["clientId"])
    else:
        X = X.loc[ixs]
        y = y.loc[ixs]

    return (X, y)


def nifti_to_data(args, X):
    """Read nifti files as matrices"""
    try:
        mask_file = os.path.join(args["state"]["baseDirectory"],
                                 'mask_6mm.nii')
        mask_data = nib.load(mask_file).get_data()
    except FileNotFoundError:
        raise Exception("Missing Mask at " + args["state"]["clientId"])

    appended_data = []

    # Extract Data (after applying mask)
    for image in X.index:
        try:
            image_data = nib.load(
                os.path.join(args["state"]["baseDirectory"],
                             image)).get_data()
            if np.all(np.isnan(image_data)) or np.count_nonzero(
                    image_data) == 0 or image_data.size == 0:
                X.drop(index=image, inplace=True)
                continue
            else:
                appended_data.append(image_data[mask_data > 0])
        except FileNotFoundError:
            X.drop(index=image, inplace=True)
            continue

    y = pd.DataFrame.from_records(appended_data)

    return X, y


def vbm_parser(args):
    """Parse the nifti (.nii) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes"""
    input_list = args["input"]
    X_info = input_list["covariates"]

    X_data = X_info[0][0]
    X_labels = X_info[1]

    X_df = pd.DataFrame.from_records(X_data)
    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))
    X_df.set_index(X_df.columns[0], inplace=True)

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X = X * 1

    X.dropna(axis=0, how='any', inplace=True)

    X, y = nifti_to_data(args, X)

    y.columns = ['{}_{}'.format('voxel', str(i)) for i in y.columns]

    return (X, y)


def main():
    print('Contains parsing functions')
    

if __name__ == '__main__':
    main()
