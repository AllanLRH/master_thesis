#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig
import collections
from speclib import misc
import itertools
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
    nxGraph : networkx.Graph or networkx.DiGraph
        Graph to convert.

    Returns
    -------
    igraph.Graph
        Converted graph, unweighted if input was a nx.Graph, weighted otherwise.
    """
    # Get adjacency matrox for networkx graph
    nxAdj = np.array(nx.adjacency_matrix(nxGraph).todense())
    # Use the binary adjacency matrix to construct the igraph graph
    if isinstance(nxGraph, nx.DiGraph):
        igGraph = adjmat2igraph(nxAdj, directed=True, labels=list(nxGraph.nodes()))
    else:
        igGraph = adjmat2igraph(nxAdj, directed=False, labels=list(nxGraph.nodes()))
    return igGraph


def adjmat2igraph(m, directed=True, labels=None):
    """Convert an numpy adjacency matrix into an weighted igraph graph.

    Parameters
    ----------
    m : np.array
        Adjacency matrix, must be positive.
    directed : bool, optional
        Construct a directed graph, default True.
    labels : list, optional
        List of labels to use. Default is strings from 0 to "number of nodes".

    Returns
    -------
    igraph.Graph
        The weighted graph.

    Raises
    ------
    ValueError
        If the input isn't a 2d-matrix
        If the adjacency matrix isn't square
        If there's elements < 0 in the adjacency matrix
    """
    if m.ndim != 2:
        raise ValueError(f"Input must be a matrix, but m.ndim = {m.ndim}.")
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"Input must be a square matrix, but m.shape = {m.shape}.")
    if np.any(m < 0):
        raise ValueError(f"all entries in the adjacency matrix must be larger than 0 ({(m < 0).sum()} < 0)")

    if directed:
        g = ig.Graph.Adjacency((m > 0).tolist(), mode=ig.ADJ_DIRECTED)
    else:
        g = ig.Graph.Adjacency((m > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    # assign weights from the adjacency matrix to the igraph graph
    g.es['weight'] = m[m.nonzero()]
    # Assign node names from the networkx graph to the igraph graph
    if labels:
        g.vs['label'] = labels
    else:
        g.vs['label'] = [str(i) for i in range(m.shape[0])]  # Give integer labels
    return g


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


def vec2squareMat(v, addDiagonal=False):
    """Reshapes a vector into a square matrix by "unstacking" the columns.

    Parameters
    ----------
    v : np.array
        Array to reshape, must have dimmension 1.
    addDiagonal : bool, optional
        The input vector doesn't include the diagonal of the vector from which it was
        created, so a 0-diagonal is added. Default False.

    Returns
    -------
    np.array
        Square matrix.

    Raises
    ------
    ValueError
        If the vector size is not compatible with a square matrix.
    """
    if not addDiagonal:
        matSize = np.sqrt(v.size)
        if not np.allclose(matSize, np.round(matSize)):
            raise ValueError(f"The size of the vector ({v.size})" +
                             " are not compatible with a square matrix.")
        matSize = int(matSize)
        return v.reshape(matSize, -1).T
    else:
        l = v.size  # noqa
        # n - sqrt(n) == l solver for n, and taking the square root to get the length of
        # a row / column, e.g. sqrt(9) = 3, and a square matrix with 9 elemens needs to
        # be of size (3, 3)
        matSizeFunc = lambda l: np.sqrt((1 + 2*l)/2 + np.sqrt(1+4*l)/2)
        matSize = matSizeFunc(l)
        if not np.allclose(matSize, np.round(matSize)):
            raise ValueError(f"The size of the vector ({v.size}) are not compatible with" +
                             " a square matrix (without the diagonal).")
        matSize = int(matSize)
        mat = np.zeros((matSize, matSize))
        i = 0
        for c in range(matSize):
            for r in range(matSize):
                if r == c:
                    continue
                mat[r, c] = v[i]
                i += 1
        return mat


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


def swapRowColIdx(m, i0, i1, inplace=False):
    """swap col and row of matrix.

    Parameters
    ----------
    m : ndarray
        Matrix for swapping
    i0 : int
        First index for row/column
    i1 : int
        Second index for row/column
    inplace : bool, optional
        Do the swapping inplace, default False

    Returns
    -------
    ndarray or None
        Returns the permuted matrix, or None if oermutation is done inplace.
    """
    if inplace:
        misc.swapMatrixCols(m, i0, i1, inplace=inplace)
        misc.swapMatrixRows(m, i0, i1, inplace=inplace)
        return None
    # Don't permute inplace
    mt = misc.swapMatrixCols(m, i0, i1)
    mt = misc.swapMatrixRows(mt, i0, i1)
    return mt


def genAllMatrixPermutations(m, dst=None):
    """Generate all row-column permutations of a graph.

    Parameters
    ----------
    m : np.array
        Adjacency matrix.
    dst : np.array, optional
        Write permuted matrix to this variable instead of including it in the yield for
        every iteration.

    Yields
    ------
    (tuple, np.array) or (tuple, None)
        permutations and permuted matrix, or if the dst-option is used, the permuted
        array is replaced with None.

    Raises
    ------
    ValueError
        If the input isn't a square matrix.
    """
    if m.shape[0] != m.shape[1]:
        raise ValueError("The functions only accepts square matrices")
    s = m.shape[0]
    mc0 = m.copy()
    if dst is None:
        mc1 = m.copy()
    for p in itertools.permutations(list(range(s))):
        # Swap rows
        np.copyto(mc0, m[p, :])
        # Swap columns
        if dst is None:
            np.copyto(mc1, mc0[:, p])
            yield (p, mc1.copy())
        else:
            np.copyto(dst, mc0[:, p])
            yield (p, None)


def dotproductGraphCompare(m0, m1):
    """Given to adjacency matrices, compare them by stacking the columns in the matrices,
    and taking the dotProduct. But do it for all row-column permutations for m0.
    The largest dot product is returned.

    Parameters
    ----------
    m0 : np.array
        Adjacency matrix which is permuted before dotting.
    m1 : np.array
        Adjacency matrix which is just used in the dot product.

    Returns
    -------
    float
        The largest of all the dot-procucts.

    Raises
    ------
    ValueError
        If m0 and m1 isn't matrices.
        If the dimmensions of m0 and m1 differ.
        If the m0 and m1 isn't square matrices.
    """
    if m0.ndim != 2 and m1.ndim != 2:
        raise ValueError(f"Input must be a matrix, bit has dimmension {m0.ndim} and {m1.ndim}")
    if m0.shape != m1.shape:
        raise ValueError(f"m0 and m1 must be same size, but m0.size = {m0.size} and m1.size = {m1.size}")
    if m0.shape[0] != m0.shape[1] or m1.shape[0] != m1.shape[1]:
        raise ValueError(f"Matrices must be quadratic, but m0.shape = {m0.shape} and m1.shape = {m1.shape}")
    v = misc.stackColumns(m1)
    dotProduct = -np.inf
    m = np.zeros_like(m0)
    for _ in genAllMatrixPermutations(m0, dst=m):
        vp = misc.stackColumns(m)
        dp = np.dot(vp, v)
        dotProduct = dp if dp > dotProduct else dotProduct
    return dotProduct
