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


def plotPunchcard(data):
    fig, ax = plt.subplots()
    pc = ax.pcolorfast(data, cmap=mpl.cm.viridis)
    fig.colorbar(pc)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    tickmarks = np.arange(0, 7*24, 24) + 12
    ax.set_xticks(tickmarks)
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45)
    ax.grid()
    for idx in np.arange(24, 7*24, 24):
        ax.axvline(x=idx, linestyle='--', color='white', linewidth=1.5, zorder=3)
    ax.set_yticklabels([])
    ax.set_ylabel('Users')
    return fig, ax
