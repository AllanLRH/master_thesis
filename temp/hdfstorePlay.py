#!/usr/bin/env ipython
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from hashlib import md5
import os


hdfName = 'storePlay.h5'

if os.path.isfile(hdfName):
    os.remove(hdfName)

str2key = lambda s: md5(s.encode('utf-8')).hexdigest()

with pd.HDFStore(hdfName) as store:
    for i in range(10, 101, 10):
        res = np.arange(i)
        resSave = pd.Series(res)
        store[str2key(str(i))] = resSave


with pd.HDFStore(hdfName) as store:
    for i, k in enumerate(store.keys()):
        # print(k, store[k])
        print(i, k, sep=':\t')

