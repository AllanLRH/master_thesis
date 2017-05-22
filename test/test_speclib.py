#!/usr/bin/env python
# -*- coding: utf8 -*-

import pytest
import pandas as pd
import numpy as np
import sys
import os
import inspect
import itertools  # noqa
import networkx as nx
import igraph as ig
from random import choice
from io import StringIO
a = inspect.currentframe()
b = inspect.getfile(a)
c = os.path.abspath(b)
d = os.path.dirname(c)
e = os.path.split(d)
sys.path.append(e[0])
from speclib import misc, graph, plotting, loaders, userActivityFunctions  # noqa
from pudb import set_trace


def test_mutualContact():
    df = pd.DataFrame(data=[['u01', 'sms', 'u42'],
                            ['u01', 'sms', 'u03'],
                            ['u02', 'sms', 'u42'],
                            ['u03', 'call', 'u42'],
                            ['u04', 'sms', 'u03'],
                            ['u03', 'call', 'u04']],
                      columns=['user', 'comtype', 'contactedUser'])
    df = df.set_index(['user', 'comtype'])

    assert userActivityFunctions.mutualContact(df, 'u01', 'u02') == False  # No contact  # noqa
    assert userActivityFunctions.mutualContact(df, 'u01', 'u03') == False  # One way contact  # noqa
    assert userActivityFunctions.mutualContact(df, 'u04', 'u03') == True  # Bidirectional contact  # noqa


def f(x, y):
    return int(x**2 + y)


def test_mapAsync():  # noqa
    itr = list(zip(range(10), range(10, 101, 10)))
    trueVal = [f(*tup) for tup in itr]
    res = misc.mapAsync(f, itr, n=2)
    assert trueVal == res


def test__isSubCommunity():
    lstDf = [list('ABCD'), list('BCDE'), list('ABC'), list('ADE'), list('ABE')]
    df = pd.DataFrame(lstDf)
    comsize = df.count(axis=1)
    bigSetLst = df[comsize == 4].apply(lambda row: set(row.dropna()), axis=1).tolist()
    smallDfRows = df[comsize < 4]
    res = graph._isSubCommunity(bigSetLst, smallDfRows)
    # print('\n')
    # print('df', df, end='\n\n', sep='\n---------------\n')
    # print('bigSetLst', bigSetLst, end='\n\n', sep='\n---------------\n')
    # print('smallDfRows', smallDfRows, end='\n\n', sep='\n---------------\n')
    # print('res', res, end='\n\n', sep='\n---------------\n')
    assert res == {3: [{'A', 'D', 'E'}, {'A', 'B', 'E'}]}


@pytest.mark.skip(message='Not finished yet')
def test_removeSubCommunitiesDumb():
    lstDf = [list('ABCD'), list('BCDE'), list('ABC'), list('ADE'),
             list('ABE'), list('AB'), list('AE'), list('BF')]
    df = pd.DataFrame(lstDf)
    new = graph.removeSubCommunitiesDumb(df)
    lstDfExpected = [list('ABCD'), list('BCDE'), list('ADE'), list('ABE'), list('BF')]
    dfExpected = pd.DataFrame(lstDfExpected)
    # print('\n')
    # print(df, end='\n-------------------------------\n')
    # print(new)
    assert thf.isArrRowSetEqual(new, dfExpected)


def test_adjMatUpper2array():
    m = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]])
    res = graph.adjMatUpper2array(m)
    assert np.all(res == np.array([1, 2, 3]))


def test_upperTril2adjMat():
    m = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]])
    u = np.array([1, 2, 3])
    res = graph.upperTril2adjMat(u)
    assert np.all(res == m)


def test_adjMatUpper2array_and_upperTril2adjMat():
    symmetricMat = np.array([[0, 1, 2, 4, 7],
                             [1, 0, 3, 5, 8],
                             [2, 3, 0, 6, 9],
                             [4, 5, 6, 0, 10],
                             [7, 8, 9, 10, 0]])
    tmp0 = graph.adjMatUpper2array(symmetricMat)
    tmp1 = graph.upperTril2adjMat(tmp0)
    assert np.all(symmetricMat == tmp1)


def test_upperTril2adjMat_and_adjMatUpper2array_with_chararray():
    for key in [list('ABCD'), ('u0676', 'u0993', 'u0618', 'u0388', 'u0645', 'u0446', 'u0683')]:
        n = len(key)
        mat = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mat[i, j] = set([key[i], key[j]])  # use sets because order is irrelevant
        upperTril = graph.adjMatUpper2array(mat)
        restored = graph.upperTril2adjMat(upperTril)
        assert np.all(restored == mat)


@pytest.fixture
def userDf2nxGraphDataFrame():
    csv = """user,comtype,contactedUser
             # u1
             u1,sms,u2
             u1,sms,u2
             u1,call,u2
             u1,sms,u3
             u1,call,u3
             u1,call,u4
             # u2
             u2,sms,u3
             u2,sms,u4
             u2,sms,u1
             # u3
             u3,call,u5
             u3,call,u4
             u3,call,u4
             u3,sms,u4
             u3,call,u5
             # u4
             u4,call,u1
             u4,call,u1
             u4,call,u1
             u4,sms,u5
             # u5
             u5,call,u4
             u5,sms,u4
             u5,sms,u4"""
    csv = StringIO('\n'.join(ln.strip() for ln in csv.splitlines()
                             if not ln.strip().startswith('#')))
    df = pd.DataFrame.from_csv(csv)
    return df


def networkEdgesSort(edgeLst):
    sortKey = lambda edge: str(edge[0]) + str(edge[1])
    return sorted(edgeLst, key=sortKey)


def test_userDf2nxGraph_Graph(userDf2nxGraphDataFrame):
    df = userDf2nxGraphDataFrame
    g = graph.userDf2nxGraph(df, graphtype=nx.Graph)
    gEdgesExpected = [('u1', 'u2', {'weight': 4}),
                      ('u1', 'u3', {'weight': 2}),
                      ('u1', 'u4', {'weight': 4}),
                      ('u2', 'u3', {'weight': 1}),
                      ('u2', 'u4', {'weight': 1}),
                      ('u3', 'u4', {'weight': 3}),
                      ('u3', 'u5', {'weight': 2}),
                      ('u4', 'u5', {'weight': 4})]
    gEdgesExpected = networkEdgesSort(gEdgesExpected)
    gEdgesActual = networkEdgesSort(g.edges(data=True))
    assert gEdgesExpected == gEdgesActual


def test_userDf2nxGraph_DiGraph(userDf2nxGraphDataFrame):
    df = userDf2nxGraphDataFrame
    g = graph.userDf2nxGraph(df, graphtype=nx.DiGraph)
    gEdgesExpected = [('u1', 'u2', {'weight': 3}),
                      ('u1', 'u3', {'weight': 2}),
                      ('u1', 'u4', {'weight': 1}),
                      ('u2', 'u1', {'weight': 1}),
                      ('u2', 'u3', {'weight': 1}),
                      ('u2', 'u4', {'weight': 1}),
                      ('u3', 'u5', {'weight': 2}),
                      ('u3', 'u4', {'weight': 3}),
                      ('u4', 'u1', {'weight': 3}),
                      ('u4', 'u5', {'weight': 1}),
                      ('u5', 'u4', {'weight': 3})]
    gEdgesExpected = networkEdgesSort(gEdgesExpected)
    gEdgesActual = networkEdgesSort(g.edges(data=True))
    assert gEdgesExpected == gEdgesActual


@pytest.mark.slow()
def test_userDf2nxGraph_Graph_random_data():
    """
    Try to make the graph construction crash
    """
    for N, nUsers in zip([2, 10, 100, 500, 2000, 5000], [2, 4, 10, 2, 3000, 2300]):
        for _ in range(30):
            fstr = 'u{:0%dd}' % len(str(nUsers))
            us = [fstr.format(i) for i in range(nUsers)]
            data = list()
            while len(data) < N:
                u = choice(us)
                uss = list(set(us) - {u, })
                ch = choice(uss)
                tp = choice(['call', 'sms'])
                data.append([u, tp, ch])
            df = pd.DataFrame(data, columns='user comtype contactedUser'.split())
            df = df.set_index(['user', 'comtype'], drop=False)
            graph.userDf2nxGraph(df)


def test_networkx2igraph():
    n = 10
    p = 0.65
    nxm = (np.random.rand(n, n) > p) * np.random.randint(0, 100, (n, n))
    np.fill_diagonal(nxm, 0)
    nxg = nx.from_numpy_matrix(nxm, create_using=nx.DiGraph())
    igg = graph.networkx2igraph(nxg)
    igm = np.array(igg.get_adjacency(attribute='weight').data)
    assert np.allclose(igm, nxm)


def test_networkx2igraph_2():
    adjmat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.15242509441228824, 0.020137859867919047],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.3307617816934398, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.11917719307637024, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.20086614426486896, 0.061391688291027646, 0.0012991302827179624],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    nxg = nx.from_numpy_matrix(adjmat, create_using=nx.DiGraph())
    nxg_adjmat = nx.adjacency_matrix(nxg).todense()
    assert np.allclose(nxg_adjmat, adjmat)
    igg = graph.networkx2igraph(nxg)
    igg_adjmat = np.array(igg.get_adjacency(attribute='weight').data)
    assert np.allclose(nxg_adjmat, igg_adjmat)


def test_igraph2networkx():
    n = 10
    p = 0.65
    igm = (np.random.rand(n, n) > p) * np.random.randint(0, 100, (n, n))
    igg = ig.Graph.Adjacency((igm > 0).tolist(), mode=ig.ADJ_DIRECTED)
    igg.es['weight'] = igm[igm.nonzero()]
    nxg = graph.igraph2networkx(igg)
    nxm = nx.adjacency_matrix(nxg).todense()
    assert np.allclose(igm, nxm)


def test_dotAllMatCols():
    m0 = np.array([[0, 2],
                   [1, 3]])
    m1 = np.array([[1, 3],
                   [2, 4]])
    assert graph.dotAllMatCols(m0, m1) == [(20, (0, 0)), (12, (1, 0))]
    m2 = np.array([[0, 1, 2],
                   [3, 0, 5],
                   [6, 7, 0]])
    m3 = np.array([[0, 2, 3],
                   [4, 0, 6],
                   [7, 8, 0]])
    assert len(graph.dotAllMatCols(m2, m3)) == 4
    assert graph.dotAllMatCols(m2, m3) == [(148), (0, 0),
                                           (133), (0, 1),
                                           (128), (2, 1),
                                           (61), (0, 2)]

