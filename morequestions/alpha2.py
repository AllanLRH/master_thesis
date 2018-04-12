#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
# import bottleneck as bn
import pandas as pd
import networkx as nx
# import igraph as ig
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=1.8)
# colorcycle = [(0.498, 0.788, 0.498),
#               (0.745, 0.682, 0.831),
#               (0.992, 0.753, 0.525),
#               (0.220, 0.424, 0.690),
#               (0.749, 0.357, 0.090),
#               (1.000, 1.000, 0.600),
#               (0.941, 0.008, 0.498),
#               (0.400, 0.400, 0.400)]
# sns.set_palette(colorcycle)
# mpl.rcParams['figure.max_open_warning'] = 65
# mpl.rcParams['figure.figsize'] = [12, 7]

# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)


from speclib import misc, loaders, graph

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)

# import pixiedust


datafiles = ['../../allan_data/weighted_graph_bluetooth.edgelist',
             '../../allan_data/weighted_graph_call.edgelist',
             '../../allan_data/weighted_graph_sms.edgelist']


# For calls and SMS:
# $$ w_{ij} = \frac{n_{ij}^{\alpha}}{ \sum_{ik} n_{ik}^{\alpha} } $$
#
# For Bluetooth:
# $$   w_{ij} = \frac{T_{ij}^{\alpha}}{ \sum_{ik} T_{ik}^{\alpha} } $$
#
# $$ r = t^2/s^2 $$
#
# $$ \bar{x} = \frac{\sum_{i > j} (x_i + x_j)}{2 w_{ij}} $$
#
# $$ s^2 = \frac{\sum_{i > j} w_{ij}\left( (x_i - \bar{x})^2 + (x_j - \bar{x})^2 \right) }{\sum_{i < j} 2w_{ij}}  $$
#
# $$ t^2 = \frac{\sum_{i > j} w_{ij}\left( (x_i - \bar{x}) (x_j - \bar{x}) \right) }{\sum_{i < j} 2w_{ij}} $$
#
# Construct the dataset $ x_i, x_j, w_{ij} $, where $x_i$ and $x_j$ are questionaire variable for persons $i$ and $j$, and $w_{ij}$ are the weight of their connection.

# Load questionaire
ua = loaders.Useralias()
qdf = pd.read_json('../../allan_data/RGender_.json')
qdf.index = qdf.index.map(lambda x: ua[x])

gca = nx.read_edgelist(datafiles[1], create_using=nx.DiGraph())
gcau = graph.nxDiGraph2Graph(gca)
dfca = nx.to_pandas_adjacency(gca)
gamu = np.array(nx.adjacency_matrix(gcau).todense())

qdf = qdf.reindex(list(gca.nodes))
qq = misc.QuestionCompleter(qdf)
q = qdf.alcohol_binge10__answer




print(q.notna().mean())
gca_q = gca.subgraph(q.index[q.notna()].tolist())
gca_qu = graph.nxDiGraph2Graph(gca_q)
gam_qu = np.array(nx.adjacency_matrix(gca_qu).todense())

n_alpha = 8
w = np.zeros((*gamu.shape, n_alpha))
alpha = np.linspace(0, 2, n_alpha)
N = gamu.shape[0]
for i in range(N):
    for j in range(i):
        if gamu[i, j] != 0.0:
            numerator = gamu[i, j] ** alpha
        else:
            numerator = np.zeros(n_alpha)
        denominator = sum(el ** alpha for el in gamu[i, (gamu[i, :] != 0)])
        assert np.isnan(denominator).any() == False, f"NaN values encountered in the 1st loop. (i, j) = {(i, j)}."  # noqa
        res = numerator / denominator
        assert np.isnan(res).any() == False, f"NaN values encountered in the 1st loop. (i, j) = {(i, j)}."  # noqa
        w[i, j, :] = res
        w[j, i, :] = res

alpha            = np.linspace(0, 2, n_alpha)
x_mean_numerator = 0
denominator      = 0
for i in range(gamu.shape[0]):
    for j in range(i):
        xi, xj           = q.iloc[i], q.iloc[j]
        x_mean_numerator += w[i, j, :] * (xi + xj)
        assert np.isnan(x_mean_numerator).any() == False, f"NaN values encountered in the 2nd loop. (i, j) = {(i, j)}."  # noqa
        denominator      += 2*w[i, j, :]
        assert np.isnan(denominator).any() == False, f"NaN values encountered in the 2nd loop. (i, j) = {(i, j)}."  # noqa
x_mean = x_mean_numerator / denominator
assert np.isnan(x_mean).any() == False, "NaN values encountered after the 2nd loop."  # noqa

t_sq_numerator = 0
s_sq_numerator = 0
for i in range(gamu.shape[0]):
    for j in range(i):
        xi, xj = q.iloc[i], q.iloc[j]
        t_sq_numerator += w[i, j, :] * (xi - x_mean) * (xj - x_mean)
        assert np.isnan(t_sq_numerator).any() == False, f"NaN values encountered in the 3rd loop. (i, j) = {(i, j)}."  # noqa
        s_sq_numerator += w[i, j, :] * ((xi - x_mean)**2 + (xj - x_mean)**2)
        assert np.isnan(s_sq_numerator).any() == False, f"NaN values encountered in the 3rd loop. (i, j) = {(i, j)}."  # noqa
t_sq = t_sq_numerator / denominator
assert np.isnan(t_sq).any() == False, "NaN values encountered after the 3rd loop."  # noqa
s_sq = s_sq_numerator / denominator
assert np.isnan(s_sq).any() == False, "NaN values encountered after the 3rd loop."  # noqa
r = t_sq / s_sq
