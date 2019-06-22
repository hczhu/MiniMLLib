#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Do kinds of manipulations for data rows from multiple tsv/csv files.
# Every file must have a header to name all columns.
# Rows from multiple files will be joined by the values of common columns
# If not common column, they are concatenated row by row.
#

import sys
import os
import logging
import argparse

import pandas as pd
from sklearn.linear_model import LinearRegression

gArgs = None


def init_logger():
    FORMAT = "%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s"
    logging.basicConfig(format=FORMAT, stream=sys.stderr, level=logging.INFO)
    logging.info("Got a logger.")
    return logging


def get_sep(f):
    assert f.endswith(".csv") or f.endswith(
        ".tsv"
    ), f"File name should end with .csv or .tsv. {f} is not valid."
    return "," if f.endswith(".csv") else "\t"


def parse_file_names(files):
    logging.info(f"Checking files: {files}")
    files = files.split(",")
    for f in files:
        get_sep(f)
        logging.info(f"Checking file: {f}")
        with open(f, "r") as fd:
            pass
    return files


def validate_filename(f):
    get_sep(f)
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        help="Data operations?",
        choices=["select_columns", "correlation_matrix", "linear_regression"],
        default="select_columns",
    )
    parser.add_argument(
        "--index_col",
        type=int,
        help="The column index of row identifier(index)",
        default=None,
    )
    parser.add_argument(
        "--data_type_is_string",
        type=bool,
        help="All data types are string",
        default=False,
    )
    parser.add_argument(
        "--output_file",
        type=validate_filename,
        help="The output file path.",
        default=None,
    )
    parser.add_argument(
        "--columns",
        help="""
            Operation arguments
                select_columns: comma separated colmun names
                correlation_matrix: two comma separated column name lists with a ':' between 2 lists
                linear_regression: Y_column=x1_column,x2_column,...
        """,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--join_on_columns",
        type=str,
        help="Comma separated columns to join files on.",
        default=None,
    )
    parser.add_argument(
        "files", help="Comma separated file names.", type=parse_file_names
    )
    global gArgs
    gArgs, unknows = parser.parse_known_args()
    if len(unknows) > 0:
        logging.warning(f"Unknown args: {unknows}")
    logging.info(
        f"op: {gArgs.op}; columns: {gArgs.columns}; files: {gArgs.files}."
    )


def read_files(files):
    dfs = []
    for f in files:
        logging.info(f"Reading file: {f}")
        sep = get_sep(f)
        dfs.append(
            pd.read_csv(
                f,
                sep=sep,
                index_col=gArgs.index_col,
                dtype=(str if gArgs.data_type_is_string else None),
            )
        )
        logging.info(
            f"Loaded dataframe with shape: {dfs[-1].shape}; columns: {dfs[-1].columns}"
        )
    df = dfs[0]
    if len(dfs) > 1:
        for r_df in dfs[1:]:
            overlapping_columns = list(
                set.intersection(
                    set(df.columns.values), set(r_df.columns.values)
                )
            )
            logging.info(f"Overlapping columns {overlapping_columns}")
            df = pd.merge(df, r_df, how="outer", on=overlapping_columns, suffixes=(False, False))
    logging.info(
        f"Got joined dataframe with shape: {df.shape}; columns: {df.columns}"
    )
    if gArgs.join_on_columns is not None:
        groupby_columns = gArgs.join_on_columns.split(",")
        logging.info(f"Groupying by columns {groupby_columns}")
        df = df.groupby(groupby_columns).first().reset_index()
    return df


def select_columns(df):
    selected_df = df
    if gArgs.columns is not None:
        columns = gArgs.columns.split(",")
        selected_df = df.loc[:, columns]

    if gArgs.output_file is None:
        print(selected_df)
    else:
        selected_df.to_csv(
            gArgs.output_file, sep=get_sep(gArgs.output_file), index=False
        )


def linear_regression(df):
    lr = LinearRegression(fit_intercept=True, normalize=True)
    y_column, x_columns = gArgs.columns.split("=")
    x_columns = x_columns.split(",")
    Y = df.loc[:, [y_column]]
    X = df.loc[:, x_columns]
    reg = lr.fit(X, Y)
    predicted_y = reg.predict(X)
    logging.info(
        "Predicted v.s. Real values: {}".format(
            "\n    ".join(
                [
                    f"{predicted_y[i, 0]:.1f} {Y.loc[i, y_column]:.1f} {100.0 * abs(predicted_y[i, 0] - Y.loc[i, y_column]) / max(1.0, Y.loc[i, y_column]):.0f}"
                    for i in range(Y.shape[0])
                ]
            )
        )
    )
    print(
        f"The coefficient of determination R^2 of the prediction: {reg.score(X, Y)}"
    )
    cof_x = reg.coef_[0, :]
    assert len(cof_x) == len(x_columns)
    print(
        "\n    ".join(
            [f"{y_column} ="]
            + [
                f"{cof_x[i]:.2f} * {x_columns[i]}"
                for i in range(len(x_columns))
            ]
            + [f"{reg.intercept_[0]:.2f}"]
        )
    )


def correlation_matrix(df):
    corr_ma = df.corr()
    left_columns, right_columns = "", ""

    if gArgs.columns == "*":
        left_columns = right_columns = ",".join(
            [str(col) for col in df.columns]
        )
    else:
        left_and_right = gArgs.columns.split(":")
        left_columns = left_and_right[0]
        right_columns = (
            left_and_right[1] if len(left_and_right) > 1 else left_columns
        )
    print(corr_ma.loc[left_columns.split(","), right_columns.split(",")])


def main():
    get_args()
    df = read_files(gArgs.files)
    ops = {
        "select_columns": select_columns,
        "correlation_matrix": correlation_matrix,
        "linear_regression": linear_regression,
    }
    ops[gArgs.op](df)


if __name__ == "__main__":
    init_logger()
    main()
