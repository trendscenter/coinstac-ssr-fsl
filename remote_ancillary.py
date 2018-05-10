#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""

import nibabel as nib
from nilearn import plotting
import numpy as np
import os
import pandas as pd


def get_stats_to_dict(a, *b):
    df = pd.DataFrame(list(zip(*b)), columns=a)
    dict_list = df.to_dict(orient='records')

    return dict_list


def print_beta_images(args, avg_beta_vector):
    beta_df = pd.DataFrame(avg_beta_vector)

    images_folder = args["state"]["outputDirectory"]

    mask_file = os.path.join(args["state"]["baseDirectory"], 'mask_6mm.nii')
    mask = nib.load(mask_file)

    for column in beta_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = beta_df[column]

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)

        plotting.plot_stat_map(
            clipped_img,
            output_file=os.path.join(images_folder, 'beta_' + str(column)),
            display_mode='ortho',
            colorbar=True)


def print_pvals(args, ps_global, ts_global):
    p_df = pd.DataFrame(ps_global)
    t_df = pd.DataFrame(ts_global)

    # TODO manual entry, remove later
    images_folder = args["state"]["outputDirectory"]

    mask_file = os.path.join(args["state"]["baseDirectory"], 'mask_6mm.nii')
    mask = nib.load(mask_file)

    for column in p_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() >
                 0] = -1 * np.log10(p_df[column]) * np.sign(t_df[column])

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)

#        thresholdh = max(np.abs(p_df[column]))

        plotting.plot_stat_map(
            clipped_img,
            output_file=os.path.join(images_folder, 'pval_' + str(column)),
            display_mode='ortho',
            colorbar=True)

        if 'display' in locals():
            plotting.plot_stat_map(
                clipped_img,
                cut_coords=display.cut_coords)
        else:
            display = plotting.plot_stat_map(
                clipped_img)
