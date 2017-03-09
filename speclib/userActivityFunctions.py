#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from speclib import graph
import networkx as nx


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


def userDf2CliqueDf(df, chosenUserLst, associatedUserColumn='contactedUser'):
    """Given a user DataFrame and a list of users, return a user DataFrame which only
    contains communication in between the users in the given list.

    Args:
        df (DataFrame): DataFrame, must have 'user' as an index.
        chosenUserLst (list): List with chosen users.
        associatedUserColumn (str, optional): Column name containing contacted users.

    Returns:
        DataFrame: With entries not involving users from chosenUserLst removed.
    """
    df = df.loc[chosenUserLst]
    return df[df[associatedUserColumn].isin(chosenUserLst)]


def userDf2timebinDf(df, bins):
    """Given a user DataFrame and bins, return a generator yielding the events for
    which fall in a given bin.
    Binning happens on a weekly timescale, and the smallest time unit is an hour.
    This all thuesdays in with hourly-bin 3 is combined, regardles of the actual date.

    Args
        df (DataFrame): User DataFrame.
        bins (str, int, or dict): Specification of bins...
              str:  Name of column with bin values.
              int:  Number of euqally spaced bins pr. 24 hours,
              dict: A mapping for every hour to a bin value

    Yields:
        DataFrame: With events from bin associated with current iteration.

    Raises:
        ValueError: If an invalid column name is provided in bin argument.
        ValueError: If a bins dict doesn't contain mapping for all hours.
    """
    df = df.copy()  # work in a copy of the input DataFrame

    # Handle bins argument as a column name
    if isinstance(bins, str):
        if (bins not in df.columns) or (df[bins].dtype not in ('i', 'U')):
            raise ValueError("If bins-argument is a string, it must " +
                             "point to a column with integers.")
        df['weekTimeBin'] = bins  # Just use the string for indexing

    # Handle bins argument as a dict with bin mappings
    if isinstance(bins, dict):
        hour = df.timestamp.dt.hour  # Compute the hours from the timestamps
        # Throw error if the bins dict doesn't map all hours
        if not len(set(bins.keys()).intersection(set(hour))) == hour.unique().size:
            raise ValueError("The bins-argument dict must contain mappings for all hours.")
        df['weekTimeBin'] = hour.map(bins)

    # Handle bins argument as an int with desired number of bins pr. 24 hours
    if isinstance(bins, int):  # if bins are an int, calculate the bins
        hour = df.timestamp.dt.hour  # Compute the hours from the timestamps
        df['weekTimeBin'] = np.floor(hour/(24/bins)).astype(np.int)  # Compute the bins

    for _, itrDf in df.groupby(['weekday', 'weekTimeBin'], as_index=False):
        yield itrDf


def userDf2timebinAdjMat(df, bins, chosenUserLst):
    """Given a user DataFrame, return a matrix where each column is the rows/columns
    from an adjacency matrix.

    Args:
        df (DataFrame): User DataFrame, must have 'user' as index and 'timestamp' as
                        a column.
        bins (str, int, or dict): Specification of bins...
              str:  Name of column with bin values.
              int:  Number of euqally spaced bins pr. 24 hours,
              dict: A mapping for every hour to a bin value
        chosenUserLst (iterable): Iterable (e.g. list) with user names from the index.

    Returns:
        np.array: Array where each column is the combined columns from the adjacency
                  matrix constructed from the events in a corresponding time bin.
    """
    aggLst = list()
    for itrDf in userDf2timebinDf(df, bins):
        itrG = graph.userDF2nxGraph(itrDf)
        itrAdj = nx.adj_matrix(itrG, nodelist=chosenUserLst)
        aggLst.append(itrAdj)
    toPcaRaw = np.zeros((len(chosenUserLst)**2, len(aggLst)))
    for i in range(len(aggLst)):
        toPcaRaw[:, i] = aggLst[i].todense().reshape((1, -1))
    return toPcaRaw
