#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""
import os.path

import pandas as pd


def get_stats_to_dict(a, *b):
    df = pd.DataFrame(list(zip(*b)), columns=a)
    dict_list = df.to_dict(orient='records')

    return dict_list

def persist_stats(roi_names, site_names, site_stats, persist_dirs):
    assert  len(site_names) == len(site_stats)
    for idx, site_name in enumerate(site_names):
        df = pd.DataFrame(site_stats[idx])
        covariate_labels = df.pop('covariate_labels')[0]
        col_names = df.columns.tolist()
        cols_with_list_vals=[]
        for idx, k in enumerate(df.loc[0]):
            if type(k) == type([]):
                cols_with_list_vals.append(idx)

        new_df=pd.concat([df[col_names[k]].apply(pd.Series) for k in cols_with_list_vals], axis=1)
        new_df.columns = [col_names[x]+"_"+y for x in cols_with_list_vals for y in covariate_labels]

        #Add remaining columns
        for col_idx in set(range(len(df.columns.tolist()))) - set(cols_with_list_vals):
            new_df[col_names[col_idx]] = df[col_names[col_idx]]

        #Add ROI names for row indexes
        new_df.index = roi_names

        #Save the dataframe as csv file
        for dir_name in persist_dirs:
            new_df.to_csv(os.path.join(dir_name, site_name+"_stats.csv"))


def main():
    print('Contains ancillary functions for remote computations')


if __name__ == '__main__':
    main()
