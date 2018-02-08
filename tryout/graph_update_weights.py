import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import networkx as nx
import re
import itertools

from speclib import misc, loaders

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

dct_0 = {('a', 'b'): 1, ('b', 'd'): 2, ('c', 'd'): 3}
dct_1 = {('a', 'b'): 6, ('b', 'a'): 7, ('c', 'a'): 8, ('d', 'b'): 9}

g = nx.Graph()
g.add_nodes_from(list('abcd'))
weight_pair_gen = ((t[0], t[1], w) for (t, w) in dct_0.items())
g.add_weighted_edges_from(weight_pair_gen)

print(nx.get_edge_attributes(g, 'weight'))

weight_pair_gen = ((t[0], t[1], w) for (t, w) in dct_1.items())
for u, v, w in weight_pair_gen:
    old_weight = g.get_edge_data(u, v, default={'weight': 0})['weight']
    g.add_edge(u, v, weight=old_weight + w)

edgedata = nx.get_edge_attributes(g, 'weight')
assert edgedata[('a', 'b')] == 14,  f"edgedata[('a', 'b')] = {edgedata[('a', 'b')]}, should have been 14."
assert edgedata[('b', 'd')] == 11,  f"edgedata[('b', 'd')] = {edgedata[('b', 'd')]}, should have been 11."
assert edgedata[('a', 'c')] == 8,  f"edgedata[('c', 'a')] = {edgedata[('c', 'a')]}, should have been 8."
