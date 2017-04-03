#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig
import logging


log = logging.getLogger('graph.py')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/graph.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
log.addHandler(fh)


def networkx2igraph(nxGraph):
    """Convert a Networkx graph to an Igraph graph.
    Note that labels are lost.
    Only tested for binary adjacency matrices.

    Args:
        nxGraph (networkx.Graph): Graph to convert.

    Returns:
        igraph.Graph: Converted graph.
    """
    return ig.Graph.Adjacency(
        nx.to_numpy_matrix(nxGraph).tolist()
        )  # noqa


def isSymmetric(m):
    """Check if a matrix (Numpy array) is summetric

    Args:
        m (aray): 2d array

    Returns:
        bool: True if symmetric, false otherwise.

    Raises:
        ValueError: It input doesn't have exactly 2 dimmensions.
    """
    if m.ndim != 2:
        raise ValueError("Input must have exactly 2 dimmensions.")
    return np.allclose(m, m.T)


def adjMatUpper2array(m):
    """Given an (adjacency) matrix, return the upper triangular part as a 1d array.

    Args:
        m (array): 2d array (adjacency matrix).

    Returns:
        array: 1d array with upper triangular part of matrix.

    Raises:
        ValueError: If input have wrong dimmensions.
        ValueError: If isn't a square matrix.
    """
    if m.ndim != 2:
        raise ValueError("The input m must have exactly 2 dimmensions.")
    if m.shape[0] != m.shape[1]:
        raise ValueError('The input must be a square matrix (it was {})'.format(m.shape))
    i, j = np.triu_indices_from(m, k=1)
    return m[i, j]


def upperTril2adjMat(up):
    """Given the upper triangular part of a quadratic matrix (excluding the diagonal,
    which is assumed to be 0), construct the corresponding symmetric matrix.

    Args:
        up (array): Upper triangular part of quadratic matrix

    Returns:
        array: Symmetric matrix where upper and lower parts are both occupied by `up`.
    """
    if up.ndim != 1:
        raise ValueError("The input m must have exactly 1 dimmension.")
    ad = np.zeros((up.size, up.size), dtype=up.dtype)
    ad[np.triu_indices_from(ad, +1)] = up
    ad += ad.T
    return ad


def igraph2networkx(igGraph):
    """Convert a Igraph graph to an Networkx graph.
    Only tested for binary adjacency matrices.

    Args:
        igGraph (igraph.Graph): Graph to convert.

    Returns:
        networkx.Graph: Converted graph.
    """
    return nx.from_numpy_matrix(
        np.array(
            igGraph.get_adjacency().data
            )
        )  # noqa


def _userDF2communicationDictOfDicts(df, userColumn='user',
                                     associatedUserColumn='contactedUser'):
    """
    Turn a DataFrame with user data into a dict of dicts, which is easily converted to a
    Networkx Graph or and adjacency-matrix-like DataFrame.

    Args:
        df (DataFrame): DataFrame as the one loaded by loadUsersParallel.
        userColumn (str, optional): Name of the level in the index containing,
                                    users initiating the communication.
        associatedUserColumn (str, optional): Name of the column containing users
                                              communicated to or associated with.

    Returns:
        Dict: Dict of dicts, listing connections and the numbers of events from outer
              key (user) to inner key (user).
    """
    userIndex = np.sort(df.index.get_level_values(userColumn).unique())
    communicationDct = dict()
    for userInit in userIndex:
        comCount = df.loc[userInit][associatedUserColumn].value_counts()
        communicationDct[userInit] = comCount.to_dict()
    return communicationDct


def userDF2nxGraph(df, userColumn='user', associatedUserColumn='contactedUser',
                   comtype=None, graphType=nx.Graph()):
    """Convert user DataFrame to a Networkx Graph.

    Args:
        df (DataFrame): DataFrame as the one loaded by loadUsersParallel.
        userColumn (str, optional): Name of the level in the index containing,
                                    users initiating the communication.
        associatedUserColumn (str, optional): Name of the column containing users
                                              communicated to or associated with.
        comtype (str, optional): Filter communication type.
        graphType (nx.Graph-like, optional): Graph type to create, pass an empty instance.

    Returns:
        nx.Graph: A Networkx graph.

    Deleted Parameters:
        diGraph (bool, optional): Set to True to return a DiGraph rather thatn a Graph.
    """
    if comtype is not None:
        df = df[df.index.get_level_values('comtype') == comtype]
    df = df.groupby([userColumn, associatedUserColumn]).comtype.count()
    df = pd.DataFrame(df.reset_index()).rename(columns={'comtype': 'weight'})
    g = nx.from_pandas_dataframe(df, source=userColumn, target=associatedUserColumn,
                                 edge_attr=['weight'])
    return g


def userDF2activityDataframe(df, userColumn='user', associatedUserColumn='contactedUser',
                             comtype=None):
    """Create an adjacency-matrix like DataFrame from the regular communication DataFrame.

    Args:
        df (DataFrame): DataFrame as the one loaded by loadUsersParallel.
        userColumn (str, optional): Name of the level in the index containing,
                                    users initiating the communication.
        associatedUserColumn (str, optional): Name of the column containing users
                                              communicated to or associated with.
        comtype (str, optional): Filter communication type.

    Returns:
        DataFrame: Adjacency-matrix like DataFrame
    """
    if comtype is not None:
        df = df[df.index.get_level_values('comtype') == comtype]
    communicationDct = _userDF2communicationDictOfDicts(df, userColumn=userColumn)
    activityDf = pd.DataFrame.from_dict(communicationDct, orient='index')
    activityDf.index.name = 'userInit'
    activityDf.columns.name = 'userRecv'
    activityDf.sort_index(axis=1, inplace=True)
    return activityDf


def removeSubCommunities(comDf, comSize=None):
    log.info("Entered removeSubCommunities")
    log.debug("comDf: {}".format(comDf))
    log.debug("comSize, before if statement: {}".format(comSize))
    if isinstance(comSize, str):
        comSize = comDf[comSize]
    else:
        comSize = comDf.select_dtypes(exclude=['int']).count(axis=1)
    log.debug("comSize, after if statement: {}".format(comSize))
    log.debug('comSize.max(): {}'.format(comSize.max()))
    for s in range(comSize.max()):
        log.debug("s: {}".format(s))
        for _, comBig in comDf[comSize == s].iterrows():
            big = set(comBig.dropna())
            for _, comSmall in comDf[comSize < s].iterrows():
                small = set(comSmall.dropna())
                logging.debug("small.issubset(big): {}".format(small.issubset(big)))
                if not small.issubset(big):
                    yield comSmall

