#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx
from speclib.plotting import drawWeightedGraph
mpl.style.use('ggplot')

g = nx.from_dict_of_dicts({'u0001': {'u0002': {'weight': 17}, 'u0003': {'weight': 14}, 'u0004': {'weight': 1}, 'u0005': {'weight': 102}},
                           'u0002': {'u0001': {'weight': 17}, 'u0003': {'weight': 74}, 'u0004': {'weight': 248}, 'u0005': {'weight': 40}},
                           'u0003': {'u0001': {'weight': 14}, 'u0002': {'weight': 74}, 'u0004': {'weight': 37}, 'u0005': {'weight': 11}},
                           'u0004': {'u0001': {'weight': 1}, 'u0002': {'weight': 248}, 'u0003': {'weight': 37}, 'u0005': {'weight': 23}},
                           'u0005': {'u0001': {'weight': 102}, 'u0002': {'weight': 40}, 'u0003': {'weight': 11}, 'u0004': {'weight': 23}}})

drawWeightedGraph(g, layout=nx.drawing.layout.circular_layout, edgeLabels=True)
plt.show()

