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
from pudb import set_trace  # noqa


@pytest.mark.userActivityFunctions
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


@pytest.mark.misc
def test_mapAsync():  # noqa
    itr = list(zip(range(10), range(10, 101, 10)))
    trueVal = [f(*tup) for tup in itr]
    res = misc.mapAsync(f, itr, n=2)
    assert trueVal == res


@pytest.mark.graph
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


@pytest.mark.graph
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


@pytest.mark.graph
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


@pytest.mark.graph
def test_adjmat2igraph():
    am = np.array([[0.0,         0.0,         0.0,         0.0,         0.0,         0.0        ],  # noqa
                   [7.34369e-02, 0.0,         0.0,         8.78858e-04, 0.0,         0.0        ],  # noqa
                   [2.62175e-04, 2.13876e-03, 0.0,         0.0,         0.0,         0.0        ],  # noqa
                   [0.0,         1.56248e-03, 1.14052e-03, 0.0,         1.15074e-03, 6.10565e-01],  # noqa
                   [5.90679e-02, 0.0,         0.0,         0.0,         0.0,         0.0        ],  # noqa
                   [5.99719e-03, 1.78558e-03, 0.0,         1.55493e-03, 0.0,         0.0        ]])  # noqa
    igGraph = graph.adjmat2igraph(am)
    amReturned = np.array(igGraph.get_adjacency(attribute='weight').data)
    assert np.allclose(am, amReturned)


@pytest.mark.graph
def test_networkx2igraph():
    n = 10
    p = 0.65
    nxm = (np.random.rand(n, n) > p) * np.random.randint(0, 100, (n, n))
    np.fill_diagonal(nxm, 0)
    nxg = nx.from_numpy_matrix(nxm, create_using=nx.DiGraph())
    igg = graph.networkx2igraph(nxg)
    igm = np.array(igg.get_adjacency(attribute='weight').data)
    assert np.allclose(igm, nxm)


@pytest.mark.graph
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


@pytest.mark.graph
def test_igraph2networkx():
    n = 10
    p = 0.65
    igm = (np.random.rand(n, n) > p) * np.random.randint(0, 100, (n, n))
    igg = ig.Graph.Adjacency((igm > 0).tolist(), mode=ig.ADJ_DIRECTED)
    igg.es['weight'] = igm[igm.nonzero()]
    nxg = graph.igraph2networkx(igg)
    nxm = nx.adjacency_matrix(nxg).todense()
    assert np.allclose(igm, nxm)


@pytest.mark.misc
def test_swapMatrixCols():
    m = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    # Swap column 0 and 1
    m_expected = np.array([[1, 0, 2],
                           [4, 3, 5],
                           [7, 6, 8]])
    assert np.allclose(misc.swapMatrixCols(m, 0, 1), m_expected)


@pytest.mark.misc
def test_swapMatrixRows():
    m = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    # Swap row 1 and 2
    m_expected = np.array([[0, 1, 2],
                           [6, 7, 8],
                           [3, 4, 5]])
    assert np.allclose(misc.swapMatrixRows(m, 1, 2), m_expected)


@pytest.mark.misc
def test_swapMatrixCols_inplace_true():
    m = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    # Swap column 0 and 1
    m_expected = np.array([[1, 0, 2],
                           [4, 3, 5],
                           [7, 6, 8]])
    misc.swapMatrixCols(m, 0, 1, inplace=True)
    assert np.allclose(m, m_expected)


@pytest.mark.misc
def test_swapMatrixRows_inplace_true():
    m = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    # Swap row 1 and 2
    m_expected = np.array([[0, 1, 2],
                           [6, 7, 8],
                           [3, 4, 5]])
    misc.swapMatrixRows(m, 1, 2, inplace=True)
    assert np.allclose(m, m_expected)


@pytest.mark.graph
def test_swapRowColIdx():
    m = np.array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]])
    m_expected = np.array([[ 0,  2,  1,  3],
                           [ 8, 10,  9, 11],
                           [ 4,  6,  5,  7],
                           [12, 14, 13, 15]])
    m_out = graph.swapRowColIdx(m, 1, 2)
    assert np.allclose(m_out, m_expected)


@pytest.mark.graph
def test_vec2squareMat_1():
    v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    m_expected = np.array([[0, 3, 6],
                           [1, 4, 7],
                           [2, 5, 8]])
    assert np.allclose(graph.vec2squareMat(v), m_expected)


@pytest.mark.graph
def test_vec2squareMat_2():
    v = np.arange(7)
    with pytest.raises(ValueError):
        graph.vec2squareMat(v)


@pytest.mark.graph
def test_vec2squareMat_3():
    v = np.array([0, 1, 2, 3, 4, 5])
    m_expected = np.array([[0, 2, 4],
                           [0, 0, 5],
                           [1, 3, 0]])
    # set_trace()
    m_actual = graph.vec2squareMat(v, addDiagonal=True)
    assert np.allclose(m_actual, m_expected)


@pytest.mark.graph
def test_genAllMatrixPermutations_1():
    m = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 9]])
    perms = list(graph.genAllMatrixPermutations(m))
    assert len(perms) == np.math.factorial(m.shape[0])


@pytest.mark.graph
def test_genAllMatrixPermutations_2():
    m = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 9]])
    perms = list(graph.genAllMatrixPermutations(m))
    # Ensure that the original matrix is included in the permutations
    assert np.any([np.allclose(m, p[1]) for p in perms])


@pytest.mark.graph
def test_genAllMatrixPermutations_3():
    m = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 9]])
    perms = list(graph.genAllMatrixPermutations(m))
    for perm, mat in perms:
        if perm == (1, 0, 2):
            m_expected = np.array([[5, 2, 8],
                                   [4, 1, 7],
                                   [6, 3, 9]])
            assert np.allclose(mat, m_expected)
        elif perm == (0, 2, 1):
            m_expected = np.array([[1, 7, 4],
                                   [3, 9, 6],
                                   [2, 8, 5]])
            assert np.allclose(mat, m_expected)
        elif perm == (2, 1, 0):
            m_expected = np.array([[9, 6, 3],
                                   [8, 5, 2],
                                   [7, 4, 1]])
            assert np.allclose(mat, m_expected)


@pytest.mark.graph
def test_genAllMatrixPermutations_4():
    # Test that dhe dst-keyword works as expected
    m = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 9]])
    dst = np.zeros_like(m)
    normal_gen = graph.genAllMatrixPermutations(m)
    dst_gen = graph.genAllMatrixPermutations(m, dst)
    for ng, dg in zip(normal_gen, dst_gen):
        assert ng[0] == dg[0]
        assert np.allclose(ng[1], dst)


@pytest.mark.graph
def test_genAllMatrixPermutations_5():
    m = np.arange(6).reshape((2, 3))
    with pytest.raises(ValueError):
        list(graph.genAllMatrixPermutations(m))
