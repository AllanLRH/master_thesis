#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath('../../speclib'))
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.style.use('ggplot')
import plotting

with pd.HDFStore('pca.h5') as store:
    pca = store['pca'].iloc[0]

pcaPlot = plotting.PcaPlotter(pca)

pcaPlot.plotHeatmap()
plt.show()

pcaPlot.plotStandardization()
plt.show()

for fig, ax in pcaPlot.plotGraphs():
    plt.show()
