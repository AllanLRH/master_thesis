#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig
import collections
from speclib import misc
# import logging


# log = logging.getLogger('graph.py')
# log.setLevel(logging.DEBUG)
# fh = logging.FileHandler('../logs/graph.log')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# log.addHandler(fh)


def networkx2igraph(nxGraph):
    """Convert a Networkx graph to an Igraph graph.
    Note that labels are lost.

    Parameters
    ----------
    nxGraph : networkx.Graph
        Graph to convert.

    Returns
    -------
    igraph.Graph
        Converted graph.
    """
    # Get adjacency matrox for networkx graph
    nxAdj = np.array(nx.adjacency_matrix(nxGraph).todense())
    # Use the binary adjacency matrix to construct the igraph graph
    igGraph = ig.Graph.Adjacency((nxAdj > 0).tolist(), mode=ig.ADJ_DIRECTED)
    # assign weights from the adjacency matrix to the igraph graph
    igGraph.es['weight'] = nxAdj[nxAdj.nonzero()]
    # Assign node names from the networkx graph to the igraph graph
    igGraph.vs['label'] = list(nxGraph.nodes())
    return igGraph


def isSymmetric(m):
    """Check if a matrix (Numpy array) is summetric

    Parameters
    ----------
    m : aray
        2d array

    Returns
    -------
    bool
        True if symmetric, false otherwise.

    Raises
    ------
    ValueError
    It input doesn't have exactly 2 dimmensions.
    """
    if m.ndim != 2:
        raise ValueError("Input must have exactly 2 dimmensions.")
    try:
        ans = np.allclose(m, m.T)
    except TypeError as e:
        try:
            ans = np.allclose(m.todense(), m.todense().T)
        except Exception:
            raise e
    return ans


def adjMatUpper2array(m):
    """Given an (adjacency) matrix, return the upper triangular part as a 1d array.

    Parameters
    ----------
    m : array
        2d array (adjacency matrix).

    Returns
    -------
    array
        1d array with upper triangular part of matrix.

    Raises
    ------
    ValueError
    If input isn't a square matrix.
    """
    if m.ndim != 2:
        raise ValueError("The input m must have exactly 2 dimmensions.")
    if m.shape[0] != m.shape[1]:
        raise ValueError('The input must be a square matrix (it was {})'.format(m.shape))
    i, j = np.triu_indices_from(m, k=1)
    return np.squeeze(np.array(m[i, j]))


def upperTril2adjMat(up):
    """Given the upper triangular part of a quadratic matrix (excluding the diagonal,
    which is assumed to be 0), construct the corresponding symmetric matrix.

    Parameters
    ----------
    up : array
        Upper triangular part of quadratic matrix

    Returns
    -------
    array
        Symmetric matrix where upper and lower parts are both occupied by `up`.

    Raises
    ------
    ValueError
        If input is not a valid size for constrcting a square matrix.
    """
    if up.ndim != 1:
        raise ValueError("The input m must have exactly 1 dimmension.")
    # For a a quadratic matrix of size (n x n) the number of elements above the diagonal,
    # s, must be s = (n^2 - n)/2.
    # Solving the equation for n yields n = 1/s + sqrt(1/4 + 2s)
    matSize = 1/2 + np.sqrt(1/4+2*up.size)
    if not np.allclose(matSize, round(matSize)):
        raise ValueError("The input is not a valid size, since it doesn't fit with " +
                         "an upper diagonal. (matrix size = {:.3f})".format(matSize))
    matSize = int(matSize)
    # It's a numeric matrix
    if up.dtype != 'object':
        ad = np.zeros((matSize, matSize), dtype=up.dtype)
        ad[np.triu_indices_from(ad, +1)] = up
        ad += ad.T
        return ad
    # It's an array with tuples of usernames
    ad = np.empty((matSize, matSize), dtype=up.dtype)
    ad[np.triu_indices_from(ad, +1)] = up
    adt = ad.T
    nullIdx = pd.isnull(ad)
    ad[nullIdx] = adt[nullIdx]
    return ad


def vec2squareMat(v):
    """Reshapes a vector into a square matrix by "unstacking" the columns.

    Parameters
    ----------
    v : np.array
        Array to reshape, must have dimmension 1.

    Returns
    -------
    np.array
        Square matrix.

    Raises
    ------
    ValueError
        If the vector size is not compatible with a square matrix.
    """
    matSize = np.sqrt(v.size)
    if not np.allclose(matSize, np.round(matSize)):
        raise ValueError("The size of the vector ({v.size}) are not compatible with a square matrix")
    matSize = int(matSize)
    return v.reshape(matSize, -1).T


def igraph2networkx(igGraph):
    """Convert a Igraph graph to an Networkx graph.

    Parameters
    ----------
    igGraph : igraph.Graph
        Graph to convert.

    Returns
    -------
    networkx.Graph
        Converted graph.
    """
    igAdj = np.array(igGraph.get_adjacency(attribute='weight').data)
    if isSymmetric(igAdj):
        return nx.from_numpy_matrix(igAdj, create_using=nx.Graph())
    else:
        return nx.from_numpy_matrix(igAdj, create_using=nx.DiGraph())


def _userDF2communicationDictOfDicts(df, userColumn='user',
                                     associatedUserColumn='contactedUser'):
    """
    Turn a DataFrame with user data into a dict of dicts, which is easily converted to a
    Networkx Graph or and adjacency-matrix-like DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame as the one loaded by loadUsersParallel.
    userColumn : str, optional
        Name of the level in the index containing, users initiating the communication.
    associatedUserColumn : str, optional
        Name of the column containing users communicated to or associated with.

    Returns
    -------
    Dict
        Dict of dicts, listing connections and the numbers of events from outer
        key (user) to inner key (user).
    """
    userIndex = np.sort(df.index.get_level_values(userColumn).unique())
    communicationDct = dict()
    for userInit in userIndex:
        comCount = df.loc[userInit][associatedUserColumn].value_counts()
        communicationDct[userInit] = comCount.to_dict()
    return communicationDct


def userDf2nxGraph(df, userIndexColumn='user', associatedUserColumn='contactedUser',
                   graphtype=nx.Graph, removeDegreeZero=True):
    """Convert user DataFrame to a Networkx Graph.

    Parameters
    ----------
    df : DataFrame
        DataFrame as the one loaded by loadUsersParallel.
    userIndexColumn : str, optional
        Name of the level in the index containing, users initiating the communication.
    associatedUserColumn : str, optional
        Name of the column containing users communicated to or associated with.
    graphtype : nx.Graph-like, optional
        Graph type to create.
    removeDegreeZero : bool, optional
        Remove nodes with degree 0.

    Returns
    -------
    Graph
        A Networkx graph of the type specified in graphType, default is a nx.Graph.
    """

    g = graphtype()  # instantiate graph
    # Add all users from df
    g.add_nodes_from(df.index.get_level_values(userIndexColumn).unique().tolist())
    # Loop over all users
    for usr in df.index.get_level_values(userIndexColumn).unique():
        # Get activity as {'contactedUserName': number of events}
        activity = df.loc[usr][associatedUserColumn].value_counts().to_dict()
        # Loop over events in activity-dict:
        for rec, weight in activity.items():
            # Add node if it's not allreaddy there, using the number of events as weight
            if not g.has_edge(usr, rec):
                g.add_edge(usr, rec, weight=weight)
            else:
                # If the node exists (possible for undirected graphs), just add to the weight
                g[usr][rec]['weight'] += weight

    # Remove degree zero nodes?
    if removeDegreeZero:
        nodesToRemove = [node for (node, degree) in g.degree() if degree == 0]
        g.remove_nodes_from(nodesToRemove)
    return g


def userDf2activityDataframe(df, userColumn='user', associatedUserColumn='contactedUser',
                             comtype=None):
    """Create an adjacency-matrix like DataFrame from the regular communication DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame as the one loaded by loadUsersParallel.
    userColumn : str, optional
        Name of the level in the index containing, users initiating the communication.
    associatedUserColumn : str, optional
        Name of the column containing users communicated to or associated with.
    comtype : str, optional
        Filter communication type.

    Returns
    -------
    DataFrame
        Adjacency-matrix like DataFrame
    """
    if comtype is not None:
        df = df[df.index.get_level_values('comtype') == comtype]
    communicationDct = _userDF2communicationDictOfDicts(df, userColumn=userColumn)
    activityDf = pd.DataFrame.from_dict(communicationDct, orient='index')
    activityDf.index.name = 'userInit'
    activityDf.columns.name = 'userRecv'
    activityDf.sort_index(axis=1, inplace=True)
    return activityDf


def _isSubCommunity(bigSetLst, smallDfRows):
    returnLst = list()
    smallSetLst = smallDfRows.apply(lambda row: set(row.dropna()), axis=1).tolist()
    bigLen = len(bigSetLst)
    for small in smallSetLst:
        # Consider usign itertools.takewhile insted of a while loop
        keep = True  # Flag to stop testing / break out of while loop
        i = 0  # counter for indexing into bigSetLst
        while keep and (i < bigLen):
            keep = not small.issubset(bigSetLst[i])  # check wether or not it's a subset
            i += 1
        if keep:  # append to returnLst if necessary
            returnLst.append(small)
    return {len(small): returnLst}


def removeSubCommunities(comDf, comSize=None, n=None):
    df = comDf.select_dtypes(exclude=['int'])
    if isinstance(comSize, str):
        comSize = comDf[comSize]
    else:
        comSize = df.count(axis=1)
    df = comDf.select_dtypes(exclude=['int'])
    sizeArr = np.unique(comSize)[::-1]  # clique sizes, largest first
    retDct = dict()
    retDct[sizeArr[0]] = df[comSize == sizeArr[0]].apply(lambda row: set(row.dropna()),
                                                         axis=1).tolist()
    retDct.update({k: list() for k in sizeArr[1:]})
    for i in range(sizeArr.shape[0] - 1):
        high = sizeArr[i]
        low = sizeArr[i + 1]
        smallCommunities = df[comSize == low]
        res = _isSubCommunity(retDct[high], smallCommunities)
        for k in res.keys():
            retDct[k].append(res[k])
    return retDct


def removeSubCommunitiesDumb(df):
    dct = collections.defaultdict(list)
    for _, row in df.select_dtypes(exclude=['int']).iterrows():
        comsize = row.count()
        dct[comsize].append(frozenset(row.dropna()))

    comsizeArr = np.array(sorted(dct.keys(), reverse=True))
    clearedSet = {el for el in dct[comsizeArr[0]]}
    itrClearedLst = list()

    for comsize in comsizeArr[1:]:
        for cmpsize in comsizeArr[comsizeArr > comsize]:
            for small in dct[comsize]:
                for big in clearedSet:
                    if not small.issubset(big):
                        itrClearedLst.append(tuple(small))
        for com in itrClearedLst:
            clearedSet.add(com)
        itrClearedLst = list()
    ret = pd.DataFrame((tuple(el) for el in clearedSet))
    ret = ret.iloc[np.argsort(ret.count(axis=1))[::-1]].reset_index(drop=True)
    return ret
