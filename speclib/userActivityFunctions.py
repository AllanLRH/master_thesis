#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from speclib import graph
import networkx as nx
from speclib import misc
from multiprocessing import Pool, cpu_count
import sys
from hashlib import md5
import warnings


def getComdataMean(df, dataCol, datasizeCol):
    """
    Get the mean value from the comFreq DataFrame, where the first point if the average
    of the most contacted telephone number for each person, the second number us the
    average for the second-most contacted person for all users and so on.

    Parameters
    ----------
    df : DataFrame
        DataFrame like comFreq (see userActivity.ipynb)
    dataCol : str
        Column with data source to use from df
    datasizeCol : str
        Column containing counts of unique values from dataCol

    Returns
    -------
    np.array
        Mean values of occurences for all uses most contacted numbers.
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

    Parameters
    ----------
    df : DataFrame
        DataFrame with communication events.
    binwidth : int, optional
        Bin witdh in seconds, default i 3600 (= 1 hour).

    Returns
    -------
    Array
        2D array with users on the y-axis and timebins on the x-axis.
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

    Parameters
    ----------
    df : DataFrame
        DataFrame as the on loaded by loadUsersParallel.
    user0 : str
        Username to test.
    user1 : str
        Username to test.

    Returns
    -------
    bool
        True of there's mutual contact
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

    Parameters
    ----------
    df : DataFrame
        DataFrame, must have 'user' as an index.
    chosenUserLst : list
        List with chosen users.
    associatedUserColumn : str, optional
        Column name containing contacted users.

    Returns
    -------
    DataFrame
        With entries not involving users from chosenUserLst removed.

    Raises
    ------
    ValueError
        When chosenUserLst isn't a list
    """
    if not isinstance(chosenUserLst, list):
        raise ValueError(f'Input is not list, but {type(chosenUserLst)}.')
    df = df.loc[chosenUserLst]
    return df[df[associatedUserColumn].isin(chosenUserLst)]


def userDf2weeklyTimebinDf(df, bins):
    """Given a user DataFrame and bins, return a generator yielding the events which fall
    in a given bin.
    Binning happens on a weekly timescale, and the smallest time unit is an hour.
    Ex: All thuesdays with hourly-bin 3 is combined, regardles of the actual date.

    Args
        df (DataFrame):
        bins

    Yields
    ------
    DataFrame
        With events from bin associated with current iteration.

    Raises
    ------
    ValueError
    If a bins dict doesn't contain mapping for all hours.

    Parameters
    ----------
    df : DataFrame
        User DataFrame.
    bins : (str, int, or dict):
        Specification of bins...
         - str:  Name of column with bin values.
         - int:  Number of euqally spaced bins pr. 24 hours,
         - dict: A mapping for every hour to a bin value.
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

    Parameters
    ----------
    df : DataFrame
        User DataFrame, must have 'user' as index and 'timestamp' as
        a column.
    bins : str, int, or dict
        Specification of bins...
        str:  Name of column with bin values.
        int:  Number of euqally spaced bins pr. 24 hours,
        dict: A mapping for every hour to a bin value
    chosenUserLst : iterable
        Iterable (e.g. list) with user names from the index.

    Returns
    -------
    np.array
        Array where each column is the combined columns from the adjacency
        matrix, constructed from the events in a corresponding time bin.
        Thus the units is activity on the y axis and timebins on the x axis.
    """
    aggLst = list()
    for itrDf in userDf2weeklyTimebinDf(df, bins):
        itrG = graph.userDf2nxGraph(itrDf)
        itrAdj = nx.adj_matrix(itrG, nodelist=chosenUserLst)
        aggLst.append(itrAdj)
    toPcaRaw = np.zeros((len(chosenUserLst)**2, len(aggLst)))
    for i in range(len(aggLst)):
        toPcaRaw[:, i] = aggLst[i].todense().reshape((1, -1))
    return toPcaRaw


def communityDf2PcaExplVarRatio(userDf, communityDf, bins, communitySizeUnique=None):
    """Given  DataFrame with communities...
    1) Strip communication events to the outisde of the community.
    2) Seperate the comminication into bins, merging bins on a weekly timescale.
    3) Build a matrix where each column is the stacked columns from an adjacency matrix
        from a single weekly bin (using the function userDf2timebinAdjMat).
    4) Do PCA analysis, and save the explained variance ratio for all communities, as
        well as their community size.

    Parameters
    ----------
    userDf : DataFrame
        DataFrame containing user activity.
    communityDf : DataFrame
        DataFrame with communities, where each line contains the
        user names of the members of the community (padded with NaN/None if
        necessary).
    bins : str, int, or dict
        int is number of bins pr 24 hours. See the
    communitySizeUnique : ints in iterable, optional
        Community size to analyze.

    Returns
    -------
    dict
        keys is community size, values are list with arrays containing the
        explained variance ratio from the PCA anaysis.
    """
    communityPcaDct = dict()
    communitySize = communityDf.count(axis=1)
    if communitySizeUnique is None:
        communitySizeUnique = communitySize.unique()
    for cs in communitySizeUnique:
        communityPcaDct[cs] = list()
        for community in communityDf[communitySize == cs].iloc[:, :cs].values:
            community = community.tolist()
            communitySubDf = userDf2CliqueDf(userDf, community)
            toPcaRaw = userDf2timebinAdjMat(communitySubDf, bins, community)
            pca = misc.pcaFit(toPcaRaw)
            communityPcaDct[cs].append(pca.explained_variance_ratio_)
    return communityPcaDct


def prepareCommunityRawData(userDf, communityLst, binColumn,
                            graphtype=nx.Graph, excludeDiagonal=False):
    """Construct a matrix where each column consists of the stacked columns from other
    generated adjacency matrices.

    Parameters
    ----------
    userDf : DataFrame
        DataFrame containing the user activity data.
    communityLst : List
        A List with the usernames in the community.
    binColumn : str
        The column in userDf containing the bin value for the entries.
    graphtype : nx.Graph or nx.DiGraph, optional
        Graph type to use, default nx.Graph.
    excludeDiagonal : bool, optional
        Exclude the diagonal from the stacked vectors, default False.

    Returns
    -------
    array
        The matrix to perform PCA analysis on.
    """
    # Strip communication outside of clique
    communityDf = userDf2CliqueDf(userDf, communityLst)
    uniqueBins = communityDf[binColumn].unique()
    if graphtype is nx.Graph:
        upperTrilSize = lambda communitySize: int((communitySize**2 - communitySize)//2)
        # Preallocate array for the PCA analysis
        toPcaRaw = np.zeros((upperTrilSize(len(communityLst)), uniqueBins.size))
        for i, tbin in enumerate(uniqueBins):
            # Mask out current timebin events
            mask = (communityDf[binColumn] == tbin).values
            # Construct a graph from the masked communication...
            gTbin = graph.userDf2nxGraph(communityDf[mask], graphtype=graphtype)
            # ... and get the adjacency-matrix for the graph
            # NOTE: community-argument not necessary?
            adjMatTbin = nx.adjacency_matrix(gTbin, communityLst)
            # If the matrix is symmetric,
            toPcaRaw[:, i] = graph.adjMatUpper2array(adjMatTbin)
        return toPcaRaw
    elif graphtype is nx.DiGraph:  # graphtype is nx.DiGraph
        nUsers = len(communityLst)
        nTimebins = len(uniqueBins)
        if excludeDiagonal:
            # Since we exclude the diagonal (n elelments), there's n**2 - n in a stacked
            # vector.
            toPcaRaw = np.zeros((nUsers**2 - nUsers, nTimebins))
            # Index with a boolean matrix, thus excluding the diagonal. Transpose the
            # matrix before indexing, because it would otherwise stack rows, not columns.
            # The index is generated by inverting a boolean identity matrix:
            #
            # v1 = np.ones(4)[:, None]
            # v2 = np.arange(1,5)[None, :]
            # mat = v1 @ v2
            # mat
            # array([[1., 2., 3., 4.],
            #       [1., 2., 3., 4.],
            #       [1., 2., 3., 4.],
            #       [1., 2., 3., 4.]])
            #
            # mask = ~np.eye(4, dtype=bool)
            # mask
            # array([[False,  True,  True,  True],
            #        [ True, False,  True,  True],
            #        [ True,  True, False,  True],
            #        [ True,  True,  True, False]])
            #
            # mat[mask]
            # array([2., 3., 4., 1., 3., 4., 1., 2., 4., 1., 2., 3.])
            #
            # mat.T[mask]
            # array([1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.])
            #
            stackMatColmns = lambda mat: mat.T[~np.eye(*mat.shape, dtype=np.bool)]
        else:  # Diagonal included
            toPcaRaw = np.zeros((nUsers**2, nTimebins))
            stackMatColmns = lambda mat: mat.T[np.ones(mat.size, dtype=np.bool).reshape(*mat.shape)]
        for i, tbin in enumerate(uniqueBins):
            mask = communityDf[binColumn] == tbin
            gTbin = graph.userDf2nxGraph(communityDf[mask], graphtype=graphtype)
            # NOTE: community-argument not necessary?
            adjMatTbin = nx.adjacency_matrix(gTbin, communityLst).todense()
            toPcaRaw[:, i] = stackMatColmns(adjMatTbin)
    else:
        raise ValueError(f"graphtype must be nx.Graph or nx.DiGraph, but was {type(graphtype)}.")
    return toPcaRaw


def communityDf2Pca(userDf, communityDf, binColumn, graphtype=nx.Graph,
                    excludeDiagonal=False):
    """Given a Dataframe with communities, return the fitted PCA class instance for all
    in a dict, where the keys is a tuple with the community members.

    Parameters
    ----------
    userDf : DataFrame
        DataFrame with user activity.
    communityDf : DataFrame
        DataFrame with communities. Integer columns are excluded.
    binColumn : str
        A string identifying the bin column of the DataFrame.
    graphtype : nx.Graph or nx.DiGraph, optional
        Graph type to use, default nx.Graph.
    excludeDiagonal : bool, optional
        Exclude diagonal from vectors, default False.

    Returns
    -------
    dict
        Dictionary where the keys is the tuple with the usernames in the community,
        and the values are the corresponding pca objects.
    """
    communityPcaDct = dict()  # Dict containing community: pca-object (return value)
    # Exclude column with clique size (optionally included in input)
    for _, community in communityDf.select_dtypes(exclude=['int']).iterrows():
        # list of usernames in community
        community = community.dropna().tolist()
        # Make the raw data for the PCA algorithm
        toPcaRaw = prepareCommunityRawData(userDf, community, binColumn,
                                           graphtype, excludeDiagonal)
        # Tha PCA input data is now build, so we do the PCA analysis
        pca = misc.pcaFit(toPcaRaw, performStandardization=True)
        pca.symmetric = True if graphtype is nx.Graph else False
        pca.community = tuple(community)
        communityPcaDct[tuple(community)] = pca
    return communityPcaDct


def community2Pca(userDf, community, binColumn, graphtype=nx.DiGraph,
                  excludeDiagonal=False, fitFunction=misc.pcaFit, fitFunctionKwargs=None):
    """Given a community, return the fitted PCA class instance for the community.
    Unless specified through fitFunctionKwargs, performStandardization is set to True in
    the fitFunction call.

    Parameters
    ----------
    userDf : DataFrame
        DataFrame with user activity.
    community : list
        Names of users in the community.
    binColumn : str
        A string identifying the bin column of the DataFrame.
    graphtype : nx.Graph or nx.DiGraph, optional
        Graph type to use, default nx.Graph.
    excludeDiagonal : bool, optional
        Exclude diagonal from the stacked vectors, default False.
    fitFunction : function, optional
        Function to user for the fitting, default is misc.pcaFit.
    fitFunctionKwargs : Dict, optional
        Dict with arguments to be passed to fitFunction.

    Returns
    -------
    class instance
        Class instance returned by fitFunction, default sklearn.decomposition.PCA with
        monkey-patched variables mean and std, if performStandardization is True.

    """
    toPcaRaw = prepareCommunityRawData(userDf, community, binColumn,
                                       graphtype, excludeDiagonal)
    # Tha PCA input data is now build, so we do the PCA analysis
    if fitFunctionKwargs is None:
        fitFunctionKwargs = dict()
    fitFunctionKwargs.setdefault('performStandardization', True)
    pca = fitFunction(toPcaRaw, **fitFunctionKwargs)
    pca.symmetric = True if graphtype is nx.Graph else False
    pca.community = tuple(community)
    return pca


def communityDf2PcaHdfParallel(userDf, communityDf, binColumn, storePath, n=None, chunksize=None):
    """Given a Dataframe with communities, return the fitted PCA class instance for all
    in a dict, where the keys is a tuple with the community members.

    Parameters
    ----------
    userDf : DataFrame
        DataFrame with user activity.
    communityDf : DataFrame
        DataFrame with communities. Integer columns are excluded.
    binColumn : str
        A string identifying the bin column of the DataFrame.

    Returns
    -------
    dict
        Dictionary where the keys is the tuple with the usernames in the community,
        and the values are the corresponding pca objects.

    Raises
    ------
    Warning
        If matrix is not symmetric
    """
    if n is None:
        n = 16 if 16 < cpu_count() else cpu_count() - 1
    chunksize = 3*n if chunksize is None else chunksize
    communityDfSplit = np.split(communityDf.select_dtypes(exclude=['int']),
                                np.arange(3, communityDf.shape[0], chunksize))
    with pd.HDFStore(storePath, 'w') as hdfstore:
        with Pool(processes=n) as pool:
            for i, comDf in enumerate(communityDfSplit):
                print('Processing community:\n\n', comDf)
                res = pool.starmap(communityDf2Pca, [ (userDf, row, binColumn) for row in  # noqa
                                   np.split(comDf, np.arange(1, comDf.shape[0])) ])   # noqa
                sys.stdout.flush()
                index, data = list(), list()
                tohash = ''
                for dct in res:
                    key = tuple(dct.keys())[0]
                    index.append(key)
                    data.append(tuple(dct.values())[0])
                    tohash += str(key)
                hashkey = 'var__' + md5(tohash.encode('utf-8')).hexdigest()
                hdfstore[hashkey] = pd.DataFrame(data=data, index=index, columns=['pca'])
