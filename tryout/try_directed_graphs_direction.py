#!/usr/bin/env python
# -*- coding: utf8 -*-
import networkx as nx


tmp = nx.DiGraph()
tmp.add_nodes_from(list('abcde'))
tmp.add_edge('a', 'b', weight=4)
tmp.add_edge('b', 'a', weight=3)
tmp.add_edge('a', 'c', weight=2)
tmp.add_edge('d', 'b', weight=9)
print("tmp.in_degree('a') = ", tmp.in_degree('a'))
print("tmp.in_degree('b') = ", tmp.in_degree('b'))
print("tmp.get_edge_data('a', 'b') = ", tmp.get_edge_data('a', 'b'))
print("tmp.get_edge_data('b', 'a') = ", tmp.get_edge_data('b', 'a'))
