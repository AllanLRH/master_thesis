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

import pandas as pd

try:
    with pd.HDFStore('../../datadump/pcaData.h5', 'r') as store:
        ccdf = store['ccdf']
        ccdfs = store['ccdfs']
        cliqueDf = store['cliqueDf']
        df = store['df']
        kcDf = store['kcDf']

    storePath = '../../datadump/pcaRun_cliques.h5'
    userActivityFunctions.communityDf2PcaHdfParallel(df, cliqueDf, 'tbin', storePath)

    storePath = '../../datadump/pcaRun_communities.h5'
    userActivityFunctions.communityDf2PcaHdfParallel(df, kcDf, 'tbin', storePath, n=4)

except Exception as e:
    jn = pushbulletNotifier.JobNotification(devices='phone')
    jn.send(e)
    raise e
