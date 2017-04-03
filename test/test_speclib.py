#!/usr/bin/env python
# -*- coding: utf8 -*-

import pytest
import pandas as pd
import numpy as np
import sys
import os
import inspect
a = inspect.currentframe()
b = inspect.getfile(a)
c = os.path.abspath(b)
d = os.path.dirname(c)
e = os.path.split(d)
sys.path.append(e[0])
from speclib import misc, graph, plotting, loaders, userActivityFunctions  # noqa


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





def test_upperTril2adjMat():
    m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert np.all(graph.adjMatUpper2array(m) == np.array([1, 2, 3]))


def test_adjMatUpper2array():
    m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    u = np.array([1, 2, 3])
    assert np.all(graph.upperTril2adjMat(u) == m)
