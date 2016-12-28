#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd


def getComdataMean(df, dataCol, datasizeCol):
    """
    Get the mean value from the comFreq DataFrame, where the first point if the average
    of the most contacted telephone number for each person, the second number us the
    average for the second-most contacted person for all users and so on.

    Args:
        df (DataFrame): DataFrame like comFreq (see userActivity.ipynb)
        dataCol (str): Column with data source to use from df
        datasizeCol (str): Column containing counts of unique values from dataCol

    Returns:
        np.array: Mean values of occurences for all uses most contacted numbers.
    """
    data = np.NaN * np.zeros((df.index.size, df[datasizeCol].max()))
    for i, user in enumerate(df.index):
        data[i, :df.loc[user][datasizeCol]] = df.loc[user][dataCol]
    data = pd.DataFrame(data)
    dataMean = data.mean(axis=0).values
    return dataMean
