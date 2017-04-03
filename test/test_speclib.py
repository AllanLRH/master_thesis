#!/usr/bin/env python
# -*- coding: utf8 -*-

import pytest
import pandas as pd
import numpy as np
import sys
import os
import inspect
import itertools
a = inspect.currentframe()
b = inspect.getfile(a)
c = os.path.abspath(b)
d = os.path.dirname(c)
e = os.path.split(d)
sys.path.append(e[0])
from speclib import misc, graph, plotting, loaders, userActivityFunctions  # noqa
import thf


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


def test_upperTril2adjMat():
    m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert np.all(graph.adjMatUpper2array(m) == np.array([1, 2, 3]))


def test_adjMatUpper2array():
    m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    u = np.array([1, 2, 3])
    assert np.all(graph.upperTril2adjMat(u) == m)
