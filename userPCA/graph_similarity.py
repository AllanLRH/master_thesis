#!/usr/bin/env python
# -*- coding: utf8 -*-

import platform
import sys
import os
if platform.system() == 'Darwin':
    pth = '/Users/allan/scriptsMount'
elif platform.system() == 'Linux':
    pth = '/lscr_paper/allan/scripts'
else:
    raise OSError('Could not identify system "{}"'.format(platform.system()))
if not os.path.isdir(pth):
    raise FileNotFoundError(f"Could not find the file {pth}")
sys.path.append(pth)

import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.style.use('ggplot')

import pickle

from speclib import graph


# Load the data:
thresh = 0.95  # Throshold for Explanation Power
with open('pca_result_clique.pickle', 'rb') as fid:
    res = pickle.load(fid)
    df = pd.DataFrame(res, columns=['clique', 'pca'])
    del res

df['cliquesize'] = df['clique'].apply(lambda lst: len(lst))
print(f"There's {df.pca.isnull().sum()} null-rows in the dataframe, out of {df.shape[0]} rows.")
print("Dropping rows with NaN's")
df = df.dropna()
df['ep'] = df.pca.apply(lambda pca: (np.cumsum(pca.explained_variance_ratio_) < thresh).sum())
df['epcom'] = df.ep / df.cliquesize

eps = 1e-8  # Entries in the adjacency matrices below the value is set to 0
for k in range(df.shape[0]):  # Loop over rows in df
    print(f"Processing {k+1} of {df.shape[0]}")
    ep = df.iloc[k].ep
    # chose the number of components required for a 95 % explanation power of the activity
    comp = df.iloc[k].pca.components_[:, :ep]
    comp[comp < eps] = 0

    # Fo the following:
    # 1.0 Construct and adjacency matrix for all modes in a community
    # 2.0 For all row-column permutations of  the first of the two adjacency matrices:
    # 2.1    Stack the columns of the (permuted) adjacency matrix, and compute the dot
    #        product.
    # 2.2    Write the largest dot product result to the correct index in the
    #        preallocated matrix.
    arr = np.NaN * np.ones((ep, ep))  # Preallocate array for pcolor-plotting
    # Loop over all combinations, omitting symmetric similar comparisons like (3, 6) and (6, 3)
    for i, j in ((i, j) for i in range(ep) for j in range(i)):
        # Construct adjacency matrices. Diagonals are added, as they were stripped before
        # the PCA analysis was done
        mi = graph.vec2squareMat(comp[:, i], addDiagonal=True)
        mj = graph.vec2squareMat(comp[:, j], addDiagonal=True)
        # Get the largest dot product
        dp = graph.dotproductGraphCompare(mi, mj)
        arr[i, j] = dp
        # arr[j, i] = dp

    # Plot the result
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    pc = ax.pcolorfast(arr, cmap='Blues_r')
    fig.colorbar(pc)
    ax.set_aspect('equal')
    # get nice ticks and labels
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xticks(np.arange(arr.shape[0]))
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_xticks(np.arange(arr.shape[0]) + 0.5, minor=True)
    ax.set_xticklabels(np.arange(arr.shape[0]) + 1, minor=True)
    ax.set_yticks(np.arange(arr.shape[0]) + 0.5, minor=True)
    ax.set_yticklabels(np.arange(arr.shape[0]) + 1, minor=True)
    ax.grid(True)
    for ext in ('.pdf', '.png'):
        fig.savefig('graph_similarity_plots/' +
                    f'cliquesize_{df.iloc[k].cliquesize}_row_{k}__clique_' +
                    '_'.join(df.iloc[k].clique) + '_pcolor' + ext)

    fig, ax = plt.subplots()
    ax.plot(np.nansum(arr, axis=0), label="Sum over rows")
    ax.plot(np.nansum(arr, axis=1), label="Sum over columns")
    ax.legend(loc='best')
    for ext in ('.pdf', '.png'):
        fig.savefig('graph_similarity_plots/' +
                    f'cliquesize_{df.iloc[k].cliquesize}_row_{k}__clique_' +
                    '_'.join(df.iloc[k].clique) + '_row_col_sum' + ext)
