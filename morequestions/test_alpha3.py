#!/usr/bin/env python
# -*- coding: utf8 -*-

import networkx as nx
import pytest
import numpy as np
from alpha3 import *


@pytest.mark.alpha
def test_set_w_ij_sms_call():
    g = nx.Graph()
    weighted_edges = [('A', 'D', 3), ('A', 'C', 4), ('B', 'C', 6), ('C', 'D', 5)]
    g.add_weighted_edges_from(weighted_edges)
    g = g.to_directed()

    tol = 1e-5

    alpha = np.array([0])
    set_w_ij_sms_call(g, alpha)
    assert (g['A']['C']['w_ij'] - 1/2 <= tol).all()
    assert (g['A']['D']['w_ij'] - 1/2 <= tol).all()
    assert (g['B']['C']['w_ij'] - 1 <= tol).all()
    assert (g['C']['A']['w_ij'] - 1/3 <= tol).all()
    assert (g['C']['B']['w_ij'] - 1/3 <= tol).all()
    assert (g['C']['D']['w_ij'] - 1/3 <= tol).all()
    assert (g['D']['A']['w_ij'] - 1/2 <= tol).all()
    assert (g['D']['C']['w_ij'] - 1/2 <= tol).all()

    alpha = np.array([1/2])
    set_w_ij_sms_call(g, alpha)
    assert (g['A']['C']['w_ij'] - 2/(2 + 3**(1/2)) <= tol).all()
    assert (g['A']['D']['w_ij'] - 3**(1/2)/(2 + 3**(1/2)) <= tol).all()
    assert (g['B']['C']['w_ij'] - 1 <= tol).all()
    assert (g['C']['A']['w_ij'] - 2/(2 + 3**(1/2) + 5**(1/2)) <= tol).all()
    assert (g['C']['B']['w_ij'] - 6**(1/2)/(2 + 3**(1/2) + 5**(1/2)) <= tol).all()
    assert (g['C']['D']['w_ij'] - 5**(1/2)/(2 + 3**(1/2) + 5**(1/2)) <= tol).all()
    assert (g['D']['A']['w_ij'] - 3**(1/2)/(3**(1/2) + 5**(1/2)) <= tol).all()
    assert (g['D']['C']['w_ij'] - 5**(1/2)/(3**(1/2) + 5**(1/2)) <= tol).all()

    alpha = np.array([1])
    set_w_ij_sms_call(g, alpha)
    assert (g['A']['C']['w_ij'] - 4/7 <= tol).all()
    assert (g['A']['D']['w_ij'] - 3/7 <= tol).all()
    assert (g['B']['C']['w_ij'] - 1 <= tol).all()
    assert (g['C']['A']['w_ij'] - 4/15 <= tol).all()
    assert (g['C']['B']['w_ij'] - 6/15 <= tol).all()
    assert (g['C']['D']['w_ij'] - 5/15 <= tol).all()
    assert (g['D']['A']['w_ij'] - 3/8 <= tol).all()
    assert (g['D']['C']['w_ij'] - 5/8 <= tol).all()

    alpha = np.array([3/2])
    set_w_ij_sms_call(g, alpha)
    assert (g['A']['C']['w_ij'] - 4**(3/2)/(4**(3/2) + 3**(3/2)) <= tol).all()
    assert (g['A']['D']['w_ij'] - 3**(3/2)/(4**(3/2) + 3**(3/2)) <= tol).all()
    assert (g['B']['C']['w_ij'] - 1 <= tol).all()
    assert (g['C']['A']['w_ij'] - 4**(3/2)/(4**(3/2) + 3**(3/2) + 5**(3/2)) <= tol).all()
    assert (g['C']['B']['w_ij'] - 6**(3/2)/(4**(3/2) + 3**(3/2) + 5**(3/2)) <= tol).all()
    assert (g['C']['D']['w_ij'] - 5**(3/2)/(4**(3/2) + 3**(3/2) + 5**(3/2)) <= tol).all()
    assert (g['D']['A']['w_ij'] - 3**(3/2)/(3**(3/2) + 5**(3/2)) <= tol).all()
    assert (g['D']['C']['w_ij'] - 5**(3/2)/(3**(3/2) + 5**(3/2)) <= tol).all()

    alpha = np.array([2])
    set_w_ij_sms_call(g, alpha)
    assert (g['A']['C']['w_ij'] - 16/25 <= tol).all()
    assert (g['A']['D']['w_ij'] - 9/25 <= tol).all()
    assert (g['B']['C']['w_ij'] - 1 <= tol).all()
    assert (g['C']['A']['w_ij'] - 16/77 <= tol).all()
    assert (g['C']['B']['w_ij'] - 36/77 <= tol).all()
    assert (g['C']['D']['w_ij'] - 25/77 <= tol).all()
    assert (g['D']['A']['w_ij'] - 9/34 <= tol).all()
    assert (g['D']['C']['w_ij'] - 25/34 <= tol).all()
