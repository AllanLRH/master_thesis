#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import random
# import bottleneck as bn
import pandas as pd
import networkx as nx

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
    """Compute and set w_ij weights on the graphs edges.

    Parameters
    ----------
    g : nx.DiGraph
        Graph to compute weightes from, and set_weights on.
    alpha : np.ndarray
        Numpy array with the alpha exponents.

    Raises
    ------
    ValueError
        If g is not a nx.DiGraph
    """
    # Expand dimmensions such that broadcasting can expand the exponentiated values
    # into the second dimmension. Make it a DataFrame, and assign names to the columns
    # and axis to ease data identification later on.

    if not type(g) == nx.DiGraph:
        raise ValueError(f"Input g must be a nx.DiGraph, but was {type(g)}.")

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
            if (u, v) not in seen:
                g[u][v]['w_ij'] = w_ij.loc[v]
                seen.add((u, v))


def get_q_mean(g, q):
    """Get the weighted mean values for a question (called \\bar{x} in the article, and q_mean in this code).

    Parameters
    ----------
    g : nx.Graph
        Graph to compute from.
    q : pd.Series
        A question from the qdf DataFrame.

    Returns
    -------
    np.ndaray
        Weighted mean of the question, in an array matching the corresponding alpha values.
    """
    numerator   = 0.0
    denominator = 0.0
    for i in g.nodes:
        for j in g[i]:  # w_ij should be treated as 0 when no connection exists
            if i != j:
                # This version of the code are probably a bit slover, but more readable
                # w_ij = g[i][j]['w_ij']
                # x_i  = q.loc[i]
                # x_j  = q.loc[j]
                # numerator   += w_ij*(x_i + x_j)
                # denominator += 2*w_ij
                numerator   += g[i][j]['w_ij']*(q.loc[i] + q.loc[j])
                denominator += 2*g[i][j]['w_ij']
    q_mean = numerator / denominator
    return q_mean


def get_s2_t2_r(g, q, q_mean):
    """Get the values for s**2, t**2 and r.

    Parameters
    ----------
    g : nx.Graph
        The graph to compute from.
    q : pd.Series
        Question from the qdf DataFrame.
    q_mean : float
        Weighted mean value of question.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        s**2, t**2 and r, in an array matching the corresponding alpha values.
    """
    numerator_s   = 0.0
    numerator_t   = 0.0
    denominator = 0.0
    for i in g.nodes:
        for j in g[i]:  # w_ij should be treated as 0 when no connection exists
            if i != j:
                numerator_s += g[i][j]['w_ij'] * ((q.loc[i] - q_mean)**2 + (q.loc[j] - q_mean)**2)
                numerator_t += g[i][j]['w_ij'] * ((q.loc[i] - q_mean) * (q.loc[j] - q_mean))
                denominator += 2*g[i][j]['w_ij']
    s_squared = numerator_s / denominator
    t_squared = numerator_t / denominator
    r = numerator_t / numerator_s
    return s_squared, t_squared, r
