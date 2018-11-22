import sys
import os
sys.path.append(os.path.abspath(".."))

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import manifold
from sklearn import cluster
from hdbscan import HDBSCAN
from umap import UMAP

import itertools
import numpy as np
# import bottleneck as bn
import pandas as pd
import seaborn as sns
sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=2.8)
colorcycle = [(0.498, 0.788, 0.498),
              (0.745, 0.682, 0.831),
              (0.992, 0.753, 0.525),
              (0.220, 0.424, 0.690),
              (0.749, 0.357, 0.090),
              (1.000, 1.000, 0.600),
              (0.941, 0.008, 0.498),
              (0.400, 0.400, 0.400)]
sns.set_palette(colorcycle)
mpl.rcParams['figure.max_open_warning'] = 65
mpl.rcParams['figure.figsize'] = [12, 7]
mpl.rcParams['text.usetex'] = True

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)

# from speclib import misc
from speclib import loaders
# from speclib import graph
# from speclib import plotting
# from speclib import userActivityFunctions

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)
mpl.rcParams['figure.figsize'] = [12, 7]


# ****************************************************************************
# *                            Load and clean data                           *
# ****************************************************************************
df = pd.io.pytables.read_hdf('../../allan_data/phone_df.h5', 'df')
import pickle
with open('useralias.pk', 'br') as fid:
    ua = pickle.load(fid)
phonebook = loaders.loadUserPhonenumberDict(ua)
original_number_of_rows = df.shape[0]
# Remove call to users not in phonebook.
df = df[df.number.isin(phonebook)]

# Add _contactedUser_ column and remove the _number_ column.
df['contactedUser'] = df.number.apply(lambda x: phonebook[x])
df = df.drop('number', axis=1)
# Drop early _very_ sparse data
df = df[df.timestamp >= '2013-07-01']
monthNameLookup = {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.', 5: 'May', 6: 'June',
                   7: 'July', 8: 'Aug.', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.'}

cleaned_number_of_rows = df.shape[0]
print("Discarded", (1 - cleaned_number_of_rows/original_number_of_rows)*100, r" % of data")


# ****************************************************************************
# *                   Load precalculated PCA dcompositions                   *
# ****************************************************************************
with pd.HDFStore('../../allan_data/pca_cliques_second_take_results_no_standardization.hdfstore') as store:
    pcadf = store['pcadf']


# Compute number of components required to capture 95% variance
pcadf['n95'] = pcadf.pca.map(lambda p: (np.cumsum(p.explained_variance_ratio_) < 0.95).sum() + 1)

for n_to_plot, plot_undirected in itertools.product([3, 4, 5, 6], [True, False]):
    # ****************************************************************************
    # *                    Build dataset from pca components_                    *
    # ****************************************************************************

    plot_type_name = "undirected" if plot_undirected else "directed"
    mask = (pcadf.symmetric == plot_undirected) & (pcadf.n_clique == n_to_plot)

    lst_components = list()
    lst_index = list()
    for ipca in pcadf[mask].pca:
        n_components = (np.cumsum(ipca.explained_variance_ratio_) <= 0.95).sum()
        to_append = ipca.components_[:, :n_components]
        lst_components.append(to_append)
        lst_index.append(np.arange(n_components))
    clique_mode = np.concatenate(lst_components, axis=1).T
    clique_index = np.concatenate(lst_index)


    # ****************************************************************************
    # *                              Do the plotting                             *
    # ****************************************************************************

    # ### Plot using t-SNE
    # Apply t-SNE and plot the vectors
    np.random.seed(1234567890)
    tsne = manifold.TSNE(perplexity=30)
    data_tsne_2d = tsne.fit_transform(clique_mode)  # 2D data for plotting

    fig, ax = plt.subplots()
    for i in pd.unique(clique_index):
        ax.scatter(*data_tsne_2d[clique_index == i].T, s=20, color=colorcycle[i], label=f"Component {i+1}")
    ax.legend(loc='upper right')
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/tsne_componnts_perp_30_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    pca_2d = decomposition.PCA(n_components=2)
    data_pca = pca_2d.fit_transform(clique_mode)


    # Try clustering the points with different algorithms
    fig, ax = plt.subplots()
    for i in pd.unique(clique_index):
        ax.scatter(*data_pca[clique_index == i].T, s=20, color=colorcycle[i], label=f"Component {i+1}")
    ax.legend(loc='upper left')
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/pca_componnts_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    # clstData = misc.standardizeData(clique_mode)
    clstData = clique_mode
    # #### KMeans

    kmeans = cluster.KMeans(max_iter=1000, n_jobs=16, n_clusters=n_to_plot)
    kmeans.fit(clstData)
    clst_kmeans = kmeans.fit_predict(clstData)
    print("Kmeans clustering", pd.value_counts(clst_kmeans), sep='\n')


    fig, ax = plt.subplots()
    ax.scatter(*data_tsne_2d.T, s=20, c=[colorcycle[i] for i in clst_kmeans], label=f"Component {i+1}")
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/tsne_kmeans_clusters_perp_30_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    # #### DBScan

    dbscan = cluster.DBSCAN(n_jobs=16, min_samples=3)
    dbscan.fit(clstData)
    clst_dbscan = dbscan.fit_predict(clstData)
    print("DBscan clustering", pd.value_counts(clst_dbscan), sep='\n')


    fig, ax = plt.subplots()
    for i in [0, 1]:
        ax.scatter(*data_tsne_2d[clst_dbscan == i].T, s=20, color=colorcycle[i], label=f"Cluster {i+1}")
    ax.scatter(*data_tsne_2d[clst_dbscan == -1].T, s=20, color=colorcycle[-1], label=f"No Cluster")
    ax.legend(loc='upper right')
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/tsne_dbscan_perp_30_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    # #### HDBScan

    hdbscan = HDBSCAN(min_cluster_size=5, core_dist_n_jobs=4)
    clst_hdbscan = hdbscan.fit_predict(clstData)
    print("Hdbscan clustering", pd.value_counts(clst_hdbscan), sep='\n')


    fig, ax = plt.subplots()
    for i in [0, 1]:
        ax.scatter(*data_tsne_2d[clst_hdbscan == i].T, s=20, color=colorcycle[i], label=f"Cluster {i+1}")
    ax.scatter(*data_tsne_2d[clst_hdbscan == -1].T, s=20, color=colorcycle[-1], label=f"No Cluster")
    ax.legend(loc='upper right')
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/tsne_hdbscan_perp_30_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    # #### T-sne perplexities

    for perplexity in [5, 10, 15, 20, 25, 45]:
        for method in ['exact', 'barnes_hut']:
            np.random.seed(1234567890)
            tsne = manifold.TSNE(perplexity=perplexity, method=method)
            data_tsne_2d_t_sne_var = tsne.fit_transform(clique_mode)

            fig, ax = plt.subplots()
            for i in [0, 1]:
                ax.scatter(*data_tsne_2d_t_sne_var[clst_kmeans == i].T, s=20, color=colorcycle[i], label=f"Cluster {i+1}")
            ax.scatter(*data_tsne_2d_t_sne_var[clst_kmeans == -1].T, s=20, color=colorcycle[-1], label=f"No Cluster")
            ax.legend(loc='lower right')
            ax.set_xlabel("First t-SNE dimension")
            ax.set_ylabel("Second t-SNE dimension")
            name = f"figs_script/tsne_hdbscan_perp_{perplexity}_seeded_colored_{n_to_plot}_clique_{plot_type_name}_method_{method}.pdf"
            fig.savefig(name)


    # #### Spectral clustering

    sp_clst = cluster.SpectralClustering(n_jobs=6, n_clusters=n_to_plot)
    clst_spectral = sp_clst.fit_predict(clstData)
    print("Spectral clustering", pd.value_counts(clst_spectral), sep='\n')


    fig, ax = plt.subplots()
    ax.scatter(*data_tsne_2d.T, s=20, c=[colorcycle[i] for i in clst_spectral], label=f"Component {i+1}")
    ax.set_xlabel("First t-SNE dimension")
    ax.set_ylabel("Second t-SNE dimension")
    fig.savefig(f"figs_script/tsne_spectralClustering_perp_30_seeded_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


    # ****************************************************************************
    # *                       UMAP dimensionality reduction                      *
    # ****************************************************************************

    for repulsion_strength in [2, 5, 10, 15, 22, 35, 50, 80, 120]:
        np.random.seed(1234567890)
        umap = UMAP(repulsion_strength=10)
        data_umap_2d = umap.fit_transform(clique_mode)

        fig, ax = plt.subplots()
        ax.scatter(*data_umap_2d.T, s=20, c=[colorcycle[i] for i in clst_kmeans], label=f"Component {i+1}")
        ax.set_xlabel("First UMAP dimension")
        ax.set_ylabel("Second UMAP dimension")
        fig.savefig(f"figs_script/umap_repulsion_{repulsion_strength}_spectralClustering_seeded_colored_{n_to_plot}_clique_{plot_type_name}.pdf")


# Thoughts…
#
# * UMAP should be included, with some references, but no in-depth explanation
# * Clustering on UMAP reduced dataset, and talk about curse of dimmensionality
# * Quote Ronald Coase, How to Lie with Statistics: "if you torture the data long enough, it will confess to anything"

# np.random.seed(1234567890)
# umap_n6 = UMAP(repulsion_strength=10, n_components=2)
# clstData_n6 = umap_n6.fit_transform(clique_mode)


# hdbscan_n6 = HDBSCAN(core_dist_n_jobs=4, min_cluster_size=3)
# hdbscan.fit(data_umap_2d)
# clst = hdbscan.fit_predict(data_umap_2d)
# print("Hdbscan clustering", pd.value_counts(clst), sep='\n')

# pd.value_counts(clst)


# fig, ax = plt.subplots()
# ax.scatter(*data_umap_2d.T, s=20, c=[colorcycle[i] for i in clst], label=f"Component {i+1}")
# ax.set_xlabel("First UMAP dimension")
# ax.set_ylabel("Second UMAP dimension")

