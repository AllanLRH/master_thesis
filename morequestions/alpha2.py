#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import json
# import bottleneck as bn
import pandas as pd
import networkx as nx
from multiprocessing import Pool
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


from speclib import loaders, graph
from speclib.pushbulletNotifier import JobNotification

jn = JobNotification()

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)

# import pixiedust




datafiles = [('../../allan_data/weighted_graph_bluetooth.edgelist', 'bluetooth'),
             ('../../allan_data/weighted_graph_call.edgelist', 'call'),
             ('../../allan_data/weighted_graph_sms.edgelist', 'sms')]


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


# for col in qdf.columns:
def calculate_r(q, col, G_org, data_origin, n_alpha, permutations=0, savepath='/lscr_paper/allan/allan_data/r_values/'):
    nan_frac = q.notna().mean()
    alpha = np.linspace(0, 2, n_alpha)
    result_dct = {'alpha': alpha}
    r_format_string = "r_perm{:0%d}" % len(str(permutations))

    # Remove persons from graph which answered Null to the question, and also drop Null values from the question
    q = q.dropna()
    G = G_org.subgraph(q.index.tolist())
    count = 0
    while count <= permutations:
        print(f"Processing {col}, count {count} of {permutations}.")
        Gu = graph.nxDiGraph2Graph(G)
        if count > 0:
            # Modifies Gu in-place
            # Keep swapping edges, one at a time, until it fails or the edges are swapped N_edges times.
            i = 0
            while i < Gu.number_of_edges():
                try:
                    nx.algorithms.swap.double_edge_swap(Gu, nswap=1)
                    i += 1
                except nx.NetworkXError:
                    break
        amca = np.array(nx.adjacency_matrix(Gu).todense())
        w = np.zeros((*amca.shape, n_alpha))
        N = amca.shape[0]
        for i in range(N):
            for j in range(i):
                if amca[i, j] != 0.0:
                    numerator = amca[i, j] ** alpha
                else:
                    numerator = np.zeros(n_alpha)
                denominator = sum(el ** alpha for el in amca[i, (amca[i, :] != 0)])
                res = numerator / denominator
                w[i, j, :] = res
                w[j, i, :] = res

        alpha            = np.linspace(0, 2, n_alpha)
        x_mean_numerator = 0
        denominator      = 0
        for i in range(amca.shape[0]):
            for j in range(i):
                xi, xj           = q.iloc[i], q.iloc[j]
                x_mean_numerator += w[i, j, :] * (xi + xj)
                denominator      += 2*w[i, j, :]
        x_mean = x_mean_numerator / denominator

        t_sq_numerator = 0
        s_sq_numerator = 0
        for i in range(amca.shape[0]):
            for j in range(i):
                xi, xj = q.iloc[i], q.iloc[j]
                t_sq_numerator += w[i, j, :] * (xi - x_mean) * (xj - x_mean)
                s_sq_numerator += w[i, j, :] * ((xi - x_mean)**2 + (xj - x_mean)**2)
        t_sq = t_sq_numerator / denominator
        s_sq = s_sq_numerator / denominator
        r = t_sq / s_sq
        result_dct[r_format_string.format(count)] = r
        count += 1
    df = pd.DataFrame(result_dct)
    df.to_msgpack(savepath + col + '_' + data_origin + '.msgpack')
    return (col, nan_frac)


# Load questionaire
ua        = loaders.Useralias()
qdf       = pd.read_json('../../allan_data/RGender_.json')
qdf.index = qdf.index.map(lambda x: ua[x])

try:
    n_alpha = 201
    N_perm = 1  # FIXME: Only using 1 permutation for teseting purposes
    with Pool(6) as pool:
        # Load graph
        for datafile, origin in datafiles:
            G_org = nx.read_edgelist(datafile, create_using=nx.DiGraph())
            # Remove persons from questionaire which ins't represneted in the graph
            qdf = qdf.reindex(list(G_org.nodes))
            # Only keep the '__answer'-columns
            qdf = qdf.filter(regex='__answer$')

            arg_generator = ((qdf[col], col, G_org, origin, n_alpha, ) for col in qdf.columns[:6])
            res = pool.starmap(calculate_r, arg_generator)
            r_dct = dict(res)

    with open("../../allan_data/r_values/nanfrac.json", 'w') as fid:
        json.dump(r_dct, fid)
except Exception as err:
    jn.send(err)
else:
    jn.send(message="Computation have finished.")