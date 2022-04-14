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
                # skipping files with repeated brain regions
                if any([count != 1 for count in y_.index.value_counts()]): continue
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

    X_df = pd.DataFrame.from_dict(X_info).T

    X = X_df.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X = X * 1

    y_labels = y_info[0]["value"]
    y_files = X.index

    y = parse_for_y(args, y_files, y_labels)

    ixs = X.index.intersection(y.index)

    if ixs.empty:
        raise Exception('No common X and y files at ' +
                        args["state"]["clientId"])
    else:
        X = X.loc[ixs]
        y = y.loc[ixs]

    return (X, y)

