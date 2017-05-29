#!/usr/bin/env ipython
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
from speclib import userActivityFunctions
from speclib import pushbulletNotifier
from speclib import loaders
from speclib import graph

import pickle
from hashlib import md5
import pandas as pd
import numpy as np
import networkx as nx
# from pudb import set_trace
from multiprocessing import Pool

str2key = lambda s: md5(s.encode('utf-8')).hexdigest()

jn = pushbulletNotifier.JobNotification(devices='phone')

try:

    df = pd.io.pytables.read_hdf('phone_df.h5', 'df')
    with open('useralias.pk', 'br') as fid:
        ua = pickle.load(fid)
    phonebook = loaders.loadUserPhonenumberDict(ua)

    # Remove data preceding this date (only use part with high activity)
    df = df[df.timestamp > '2013-08-21']

    # Remove call to users not in phonebook.
    df = df[df.number.isin(phonebook)]

    # Add _contactedUser_ column and remove the _number_ column.
    df['contactedUser'] = df.number.apply(lambda x: phonebook[x])
    df = df.drop('number', axis=1)

    # ## Remove entries with users contacting themself
    tmp = df.reset_index()
    tmp = tmp[(tmp.user != tmp.contactedUser)]
    df = tmp.set_index(['user', 'comtype'], drop=False)
    del tmp

    # # Turn data into a Networkx graph.
    # Require mutual contact inititive for all nodes for a connection to exist in the
    # undirected graph g
    cdf = df.xs('call', level=1)
    cdf = cdf.reset_index(drop=True).set_index(['user', 'comtype'], drop=False)
    sdf = df.xs('sms', level=1)
    sdf = sdf.reset_index(drop=True).set_index(['user', 'comtype'], drop=False)
    gc = graph.userDf2nxGraph(cdf, graphtype=nx.DiGraph)
    gs = graph.userDf2nxGraph(sdf, graphtype=nx.DiGraph)
    g = gc.to_undirected(reciprocal=True)
    g.add_nodes_from(gs.nodes())
    g.add_weighted_edges_from(gs.to_undirected(reciprocal=True).edges(data='weight'))

    # Ensure that the graph contains the correct number of nodes
    assert len(list(g.nodes())) == len(set(df.index.get_level_values('user').tolist() + df.contactedUser.tolist()))

    # # Clique detection
    cliqueDf = pd.DataFrame(nx.clique.find_cliques_recursive(g))

    cliqueDf['cliqueSize'] = cliqueDf.count(axis=1)
    cliqueDf = cliqueDf.sort_values('cliqueSize', ascending=False)
    cliqueDf = cliqueDf.reset_index(drop=True)
    cliqueDf = cliqueDf[cliqueDf.cliqueSize > 2]

    # Find communities
    communityDf = pd.DataFrame(sorted(nx.algorithms.community.k_clique_communities(g, 5),
                                      key=lambda x: len(x), reverse=True))
    communityDf.columns.name = 'users'
    communityDf.index.name = 'communityNumber'

    # ### Determine start time offset for the binning
    # Find the first occuring communication...
    t0 = df.timestamp.min()
    t0d = pd.Timestamp(t0.date())

    # Since the timeint is in seconds, but Pandas keeps it's records in nanoseconds, the
    # integer representation of the date needs to be divided by 1e9. To check that this is
    # indeed true, compare the values of the integer casted `t0` to the timeint for the
    # corresponding row:
    t0d = np.int64(t0d.value // 1e9)
    t0 = np.int64(t0.value // 1e9)

    # Binning is simply performed by integer division with a suiting bin width.
    # I choose 8 hours:
    bw8h = 60**2 * 8
    df['tbin'] = (df.timeint - t0d) // bw8h

    # Sample one clique of each size
    ccdfs = pd.DataFrame([cliqueDf[cliqueDf.cliqueSize == i].iloc[0] for i in cliqueDf.cliqueSize.unique()])
    # Perform pca analysis
    # set_trace()

    # dct = dict()
    # for com in [tuple(row.dropna().tolist()) for (i, row) in ccdfs.drop('cliqueSize', axis=1).iterrows()]:
    #     dct[com] = userActivityFunctions.communityDf2Pca(df, ccdfs, 'tbin', graphtype=nx.DiGraph)
    #     print(com, dct[com], sep='\n', end='\n------------------------\n\n')

    def handle_community2pca(com):
        try:
            ret = (com, userActivityFunctions.community2Pca(df, com, 'tbin', nx.DiGraph, True))
        except ValueError as e:
            print("An error was encountered, continueing execution")
            print(e)
            try:
                jn.send(e)
            except:  # noqa
                pass
            ret = [com, None]
        return ret

    # savename = 'pca_result_clique.pickle'
    # communityLst = [row.dropna().tolist() for (i, row) in cliqueDf.drop('cliqueSize', axis=1).iterrows()]
    # with Pool(16) as pool:
    #     call = pool.map_async(handle_community2pca, communityLst)
    #     call.wait()
    #     res = call.get()
    #     for k, v in res:
    #         print(k, v)
    #     if os.path.exists(savename) and os.path.isfile(savename):
    #         os.remove(savename)
    #     with open(savename, 'bw') as fid:
    #         pickle.dump(res, fid, protocol=3)

    savename = 'pca_result_community.pickle'
    communityLst = [row.dropna().tolist() for (i, row) in communityDf.iterrows()]
    with Pool(16) as pool:
        call = pool.map_async(handle_community2pca, communityLst)
        call.wait()
        res = call.get()
        for k, v in res:
            print(k, v)
        if os.path.exists(savename) and os.path.isfile(savename):
            os.remove(savename)
        with open(savename, 'bw') as fid:
            pickle.dump(res, fid, protocol=3)

except Exception as e:
    jn.send(exception=e)
    raise e
jn.send("Job's done")
