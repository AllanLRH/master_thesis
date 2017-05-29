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

from hashlib import md5
import pandas as pd
import logging


jn = pushbulletNotifier.JobNotification(devices='phone')

str2key = lambda s: md5(s.encode('utf-8')).hexdigest()

logging.basicConfig(filename='userPCA_long_time_binning_pool.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(module)s:%(filename)s:%(funcName)s:line %(lineno)d: %(message)s')

dataSource = '../../datadump/pcaData.h5'
if not os.path.isfile(dataSource):
    raise FileNotFoundError(f"The file {dataSource} was not found")
with pd.HDFStore(dataSource, 'r') as store:
    ccdf = store['ccdf']
    ccdfs = store['ccdfs']
    cliqueDf = store['cliqueDf']
    df = store['df']
    kcDf = store['kcDf']


print('ccdf', ccdf.shape, sep=':\t')
print('ccdfs', ccdfs.shape, sep=':\t')
print('cliqueDf', cliqueDf.shape, sep=':\t')
print('df', df.shape, sep=':\t')
print('kcDf', kcDf.shape, sep=':\t')

timebinColumn = 'tbin'

storePath = '../../datadump/pcaRun_cliques.h5'
logging.info(f"Starting processing of {storePath}")
with pd.HDFStore(storePath) as store:
    for r in range(cliqueDf.shape[0]):
        try:
            logging.info("Starting processing clique:\n{}".format(cliqueDf.iloc[r]))
            singleCliqueDf = cliqueDf.iloc[[r]]
            res = userActivityFunctions.communityDf2Pca(df, singleCliqueDf, timebinColumn)
            store[str2key(str(res))] = pd.Series((res, ))  # Series of a tuple
            logging.info("Processing done")
        except Exception as e:
            logging.critical(f"An exception occured: {e}. r is {r} and the correponding clique dataframe are {cliqueDf.iloc[[r]]}.")
            jn.send(e)


storePath = '../../datadump/pcaRun_communities.h5'
logging.info(f"Starting processing of {storePath}")
with pd.HDFStore(storePath) as store:
    for r in range(kcDf.shape[0]):
        try:
            logging.info("Starting processing clique: {}".format(kcDf.iloc[r]))
            singlekcDf = kcDf.iloc[[r]]
            res = userActivityFunctions.communityDf2Pca(df, singlekcDf, timebinColumn)
            store[str2key(str(res))] = pd.Series((res, ))  # Series of a tuple
            logging.info("Processing done")
        except Exception as e:
            logging.critical(f"An exception occured: {e}. r is {r} and the correponding clique dataframe are {kcDf.iloc[[r]]}.")
            jn.send(e)

jn.send()
