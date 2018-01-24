#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=1.8)
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

from speclib import misc, plotting, loaders

df = pd.read_msgpack('../../allan_data/bluetooth_light_no_nan.msgpack')

print(df.head())

print(df.shape)

df['hour'] = df.index.hour
print("Done computing hour")
df['weekday'] = df.index.weekday
print("Done computing weekday")
before_workday = df.weekday.isin({0, 1, 2, 3, 6})  # is it monday, tuesday, wendnesday, thursday or sunday?
print("Done computing before_workday")
free_time = (19 < df.hour) | (df.hour < 7)
print("Done computing free_time")

dfti = df[df.user == 'u0182'].index

dftiu = dfti.unique()
dftiu = dftiu.sort_values()

print(dftiu.dtype)

index_delta = list()
for i in range(len(dftiu) - 1):
    index_delta.append(dftiu[i+1] - dftiu[i])
index_delta = pd.Series(index_delta)

print(index_delta.describe(include='all'))


# Krav: Folk skal være sammen i mindst 2 timer før det tæller som hænge ud sammen, og deres
# signaler skal observeres i mindst 70 % af tiden før de tæller som at være sammen.

# Sample try:
#
# ```
# dfs = df.sample(4500)
# dfs = dfs[dfs.before_workday & dfs.free_time]
#
# dfs.groupby(['user', dfs.index.weekday_name]).rssi.count()
# ```
# dfs = df.sample(20)
dfs = df[before_workday & free_time]
# dfs['user_id'] = dfs.scanned_user.replace(np.NaN, dfs.bt_mac)
# dfs['scanned_user'] = dfs.scanned_user.replace(np.NaN, 'unknown')

tmp = dfs.iloc[:4000]


# Check that timestamps and timedaltas can be used for binning/slicing
print(tmp.index[0])

print(tmp.index[0] + pd.Timedelta(4, unit='h'))

print((
    ( tmp.index[0] <= tmp.index ) &
    ( tmp.index <= (tmp.index[0] + pd.Timedelta(4, unit='h')) )
)[:10])

print(dfs.head())


def concatenater(args, frac=0.70):
    vc = args.value_counts()
    return set(vc[vc >= frac*vc.max()].index)


def mostly_present_counter(args):
    return len(concatenater(args))
    # con_len = len(concatenater(args))
    # return int(con_len) if con_len else None


# Resampling `df` works, but it's not ideal since it's not organized pr. user basis
tmp2 = df.iloc[:1000][['user', 'scanned_user']].resample('2h', closed='left').agg(concatenater)
print(tmp2.head(7))


# A solution where the data is grouped pr. user basis, and thus usefor for multiprocessing
#
# ```
# dfs2 = dfs2.set_index(['user', 'timestamp'])
#
# dfs2.head(12)
#
# tmp3 = dfs2.loc['u0182'].iloc[:1000]['scanned_user'].resample('4h', closed='left').agg(concatenater)
# tmp3.head(12)
# ```
tmp3 = dfs.iloc[:3000].groupby('user')[['scanned_user']].resample('4h', closed='left').agg(concatenater)

print(tmp3.head(12))

grouped = dfs.iloc[:3000].groupby('user')[['scanned_user']].resample('4h', closed='left').agg(concatenater)

print(grouped.head(20))

grouped['scanned_user'] = grouped.scanned_user.replace(set(), np.NaN)

print(grouped.head())

print(grouped.dropna().shape)


