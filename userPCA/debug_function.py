#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
# import bottleneck as bn
import pandas as pd
import networkx as nx
# import igraph as ig
# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)


from speclib import misc, userActivityFunctions, graph

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)




with pd.HDFStore('../../allan_data/pca_data.hdf5') as store:
    df = store['df']
    cliques = store['cliques']

kwargs = dict(userDf=df, communityLst=cliques.iloc[-1].dropna().tolist(), binColumn='tbin')
mat = userActivityFunctions.prepareCommunityRawData_2(**kwargs)

print(mat)


