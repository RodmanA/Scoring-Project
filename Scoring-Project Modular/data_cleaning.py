"""
data_cleaning.py
This module contain all the function to export the data and clean them
"""

import os
import pandas as pd


def get_df_csv(filename, na_value):
    """
    Import data from csv file located in the working directory
    with the name 'filename' and the missing value 'na_value'

    returns as a pandas dataframe
    """
    base_path = os.getcwd()
    file_path = os.path.join(base_path, filename)
    print(file_path)
    df = pd.read_csv(file_path, sep=';', na_values=[na_value])
    print(f"[OK] File loaded successfully: {filename}")
    return df


def adjust_french_decimal(df):
    """
    This function adjusts the French decimal from ',' to '.'
    """
    df.replace(',', '.', regex=True, inplace=True)
    return df


def split_data(df, Yd, rows):
    """
    This function splits the dataframe into a train and test sample between odd or even rows for a given variable Xd

    returns as a pandas dataframe
    """
    if rows not in ["odd", "even"]:
        raise ValueError("rows must be 'odd' or 'even'")

    test_mask = df.index % 2 == (1 if rows == "odd" else 0)

    # Explanatory vars = all columns except target
    X_cols = [col for col in df.columns if col != Yd]

    # Split explanatory variables
    X_train = df.loc[~test_mask, X_cols].reset_index(drop=True)
    X_test = df.loc[test_mask, X_cols].reset_index(drop=True)

    # Split target
    y_train = df.loc[~test_mask, Yd].reset_index(drop=True)
    y_test = df.loc[test_mask, Yd].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
