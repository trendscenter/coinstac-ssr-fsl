import os
import pandas as pd
import utils as ut

def parse_for_y(args, y_files, y_labels):
    """Read contents of fsl files into a dataframe"""
    y = pd.DataFrame(index=y_labels)
    ut.log(f'ROIs provided {str(y)}', args["state"])

    for file in y_files:
        if file:
            try:
                y_ = pd.read_csv(
                    os.path.join(args["state"]["baseDirectory"], file),
                    sep='\t',
                    header=None,
                    names=['Measure:volume', file],
                    index_col=0)
                #ut.log(f'Before Measure Removal: {list(y_.index)}', args["state"])
                y_ = y_[~y_.index.str.contains("Measure:volume")]
                #ut.log(f'after removal : {list(y_.index)}', args["state"])
                # skipping files with repeated brain regions
                repeated_brain_regions=set(y_labels).intersection(y_[y_.index.duplicated()].index.tolist())
                if any(repeated_brain_regions):
                    ut.log(f'SKIPPING file {os.path.join(args["state"]["baseDirectory"], file)} which has repeated brain region measures for {str(repeated_brain_regions)}', args["state"])
                    continue
                #ut.log(f'before numeric after count.. {list(y_.index)}', args["state"])
                y_ = y_.apply(pd.to_numeric, errors='ignore')
                #ut.log(f'after numeric {list(y_.index)}', args["state"])
                y = pd.merge(
                    y, y_, how='left', left_index=True, right_index=True)
                ut.log(f'Processed file {os.path.join(args["state"]["baseDirectory"], file)}', args["state"])
            except pd.errors.EmptyDataError:
                ut.log(f'Empty content in the file {os.path.join(args["state"]["baseDirectory"], file)}', args["state"])
                continue
            except FileNotFoundError:
                ut.log(f'File not found{os.path.join(args["state"]["baseDirectory"], file)}', args["state"])
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

    ut.log(f'\nX index: {list(X.index)}', args["state"])
    ut.log(f'\nY index: {list(y.index)}', args["state"])

    ixs = X.index.intersection(y.index)

    if ixs.empty:
        raise Exception('No common X and y files at ' +
                        args["state"]["clientId"])
    else:
        X = X.loc[ixs]
        y = y.loc[ixs]

    return (X, y)


