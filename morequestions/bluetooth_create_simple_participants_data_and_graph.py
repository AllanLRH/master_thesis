import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
import networkx as nx
import re
import itertools
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

from speclib import misc, loaders

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


ua = loaders.Useralias()
userlist = loaders.getUserList()

g = nx.Graph()
nicenames = [ua[user] for user in userlist]
g.add_nodes_from(nicenames)

for i, userhash in enumerate(userlist):
    print(f"Processing {ua[userhash]} {i}/{len(userlist)}")
    user = ua[userhash]
    df = loaders.loadUserBluetooth(userhash, ua)
    if df is None:
        continue  # Don't process na-users
    df = df.dropna()
    df = df[df.user != df.scanned_user]  # drop users registering them selves
    df_cnt = df.scanned_user.value_counts()
    weight_gen_pair = ((user, scanned_user, weight) for (scanned_user, weight) in df_cnt.iteritems())
    for u, v, w in weight_gen_pair:
        old_weight = g.get_edge_data(u, v, default={'weight': 0})['weight']
        g.add_edge(u, v, weight=old_weight + w)

gdf = nx.to_pandas_adjacency(g, nodelist=nicenames)
# # Ensure that the diagonal are 0, which is not guaranteed with dirty data
for nd in gdf.index:
    if gdf.loc[nd, nd] != 0:
        print(gdf.loc[nd, nd].index)
    # gdf.loc[nd, nd] = 0
gdf.to_msgpack('../../allan_data/participants_graph_adjacency.msgpack')
