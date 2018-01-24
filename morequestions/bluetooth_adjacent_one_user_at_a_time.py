#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np  # noqa
import pandas as pd
import itertools
from speclib import loaders


ua = loaders.Useralias()
userlist = loaders.getUserList()

# # Goal
#
# * Filter data for omnipresent devices, which are a bad proxy for social interactions
# * Group data
#
user = userlist[0]
morning_hour = 7
evening_hour = 17


def filterUserMac(*args):
    user, ua, evening_hour, morning_hour = args
    df = loaders.loadUserBluetooth(user, ua)
    if df is None:
        return None
    # Filter out time where they're probably at DTU
    df = df[(df.index.hour > evening_hour) | (df.index.hour < morning_hour)]
    # Group to day of year
    day_of_year_grouped = df.groupby(df.index.dayofyear)['bt_mac'].unique()
    # Count occurence of bluetooth mac addresses, summed for all days (in free time)
    cnt = pd.value_counts(el for el in itertools.chain(*day_of_year_grouped))
    size_lst      = list()
    threshold_lst = list()
    for thr in cnt.unique():
        threshold_lst.append(thr)
        masked_idx = set(cnt[cnt > thr].index)
        masked_df = df[~df.bt_mac.isin(masked_idx)]
        size_lst.append(masked_df.shape[0])
    var = pd.Series(size_lst, index=threshold_lst)
    return cnt, var


res = dict()
for user in userlist:
    print("Processing user", ua[user])
    res[user] = filterUserMac(user, ua.userdct, evening_hour, morning_hour)
