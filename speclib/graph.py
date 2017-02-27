#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig


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
                   comtype=None, diGraph=False):
    """Convert user DataFrame to a Networkx Graph.

    Args:
    df (DataFrame): DataFrame as the one loaded by loadUsersParallel.
    userColumn (str, optional): Name of the level in the index containing,
                                users initiating the communication.
    associatedUserColumn (str, optional): Name of the column containing users
                                          communicated to or associated with.
    comtype (str, optional): Filter communication type.
    diGraph (bool, optional): Set to True to return a DiGraph rather thatn a Graph.

    Returns:
        nx.Graph: A Networkx graph.
    """
    if comtype is not None:
        df = df[df.index.get_level_values('comtype') == comtype]
    communicationDct = _userDF2communicationDictOfDicts(df, userColumn=userColumn)
    if diGraph:
        return nx.DiGraph(communicationDct)
    # Drop the weights: Only the keys (users communicated to) is necessary
    return nx.Graph({k: v.keys() for (k, v) in communicationDct.items()})


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
    activityDf = pd.DataFrame(communicationDct)
    activityDf.index.name = 'userInit'
    activityDf.columns.name = 'userRecv'
    return activityDf
