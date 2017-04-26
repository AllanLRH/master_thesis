#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath('..'))
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.style.use('ggplot')

from speclib import userActivityFunctions

store = pd.HDFStore('dbg.h5')
df = store['df']
ccdf = store['ccdf']

dct = userActivityFunctions.communityDf2Pca(df, ccdf, 'tbin')
