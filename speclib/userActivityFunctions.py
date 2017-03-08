#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def df2punchcard(df, binwidth=3600):
    """Make a punchcard over all time for the communication-dataframe df.
       I suspect it's buggy!

    Args:
        df (DataFrame): DataFrame with communication events.
        binwidth (int, optional): Bin witdh in seconds, default i 3600 (= 1 hour).

    Returns:
        Array: 2D array with users on the y-axis and timebins on the x-axis.
    """
    minInt, maxInt = df.timeint.min(), df.timeint.max()
    deltaInt = maxInt - minInt
    nBins = 3600  # (deltaInt % binwidth) + 1
    nUsers = len(df.index.get_level_values('user').unique())
    arr = np.zeros((nUsers, nBins))
    for i, user in enumerate(df.index.get_level_values('user').unique()):
        try:
            indexArr = df.loc[user].timeint.apply(lambda x: (x - minInt) % binwidth).as_matrix()
            arr[i, :] = np.bincount(indexArr, minlength=nBins)
        except ValueError as e:
            print(i, user, nBins, deltaInt, e, sep="\n\n")
    return arr


def mutualContact(df, user0, user1):
    """Return true if user0 nad user1 have contacted each other.

    Args:
        df (DataFrame): DataFrame as the on loaded by loadUsersParallel.
        user0 (str): Username to test.
        user1 (str): Username to test.

    Returns:
        bool: True of there's mutual contact
    """
    mutualContact = set()
    if (user0, user1) in mutualContact:
        print("looked up!")
        return True
    user01 = df.loc[user0].contactedUser.str.contains(user1).any()
    user10 = df.loc[user1].contactedUser.str.contains(user0).any()
    if (user01 and user10):
        mutualContact.add((user0, user1))
        mutualContact.add((user1, user0))
        return True
    return False
