#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
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
