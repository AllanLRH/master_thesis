#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import random
import itertools
# import bottleneck as bn
import pandas as pd
import networkx as nx
from multiprocessing import Pool

from speclib import loaders, graph, misc
from speclib.pushbulletNotifier import JobNotification


jn = JobNotification()

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)


datafiles = [('../../allan_data/weighted_graph_bluetooth.edgelist', 'bluetooth'),
             ('../../allan_data/weighted_graph_call.edgelist', 'call'),
             ('../../allan_data/weighted_graph_sms.edgelist', 'sms')]


def get_aligned_data(q, g):
    """Create version of a question and a graph, where only the
    intersection between them are kept.

    Parameters
    ----------
    q : pd.Series
        Series with question data (__answer column).
    g : nx.DiGraph
        A graph containing connections between participants.

    Returns
    -------
    (pd.Series, nx.DiGraph, float)
        A tuple with the aligned question, graph and a float indicating how
        much data that WASN'T discarded.
    """
    not_na_mask = q.notna()
    q_keep_set  = set(q.index[not_na_mask])
    g_keep_set  = set(g.nodes)
    keep_index  = q_keep_set.intersection(g_keep_set)
    q_filtered  = q.loc[keep_index]
    # NOTE: Create copy of the subgraph, so that it may be modified. Consider not to.
    g_filtered  = g.subgraph(keep_index).copy()
    return (q_filtered, g_filtered, not_na_mask.mean())


def shuffle_graph_weights(g):
    """Shuffle the weights of the links in a graph.

    Parameters
    ----------
    g : nx.Graph
        Graph to shuffle links weights in, must be stored in 'weights'.

    """
    links, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
    weights = list(weights)
    random.shuffle(weights)
    shuffled_weight_dict = {(u, v): w for ((u, v), w) in zip(links, weights)}
    nx.set_edge_attributes(g, shuffled_weight_dict, name='weight')
    # # Code for returning a new graph with shuffled wegihts
    # gg = nx.Graph()
    # gg.add_weighted_edges_from((u, v, w) for ((u, v), w) in zip(links, weights))
    # return gg


def set_w_ij_sms_call(g, alpha):
    # Expand dimmensions such that broadcasting can expand the exponentiated values
    # into the second dimmension. Make it a DataFrame, and assign names to the columns
    # and axis to ease data identification later on.

    seen = set()  # Keep track of seen pairs/edges
    for u in g.nodes:

        # Return if all pairs/edges are processed
        if len(seen) == g.number_of_edges():
            return None

        # Get weights and a list of friends
        u_weights = np.array([v['weight'] for v in g[u].values()])
        u_friends = list(g[u])

        # Construct w_ij(alpha)
        w_ij = pd.DataFrame(u_weights[:, np.newaxis]**alpha, columns=alpha, index=u_friends)
        w_ij = w_ij / w_ij.sum(axis=0)
        w_ij.columns.name = 'alpha'
        w_ij.index.name   = f'{u} friends'

        # Set the resulting edge properties
        for v in u_friends:
            if {u, v} not in seen:
                g[u][v]['w_ij'] = w_ij.loc[v]
                seen.add(frozenset(u, v))





