#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np  # noqa
import pandas as pd
import itertools
from speclib import loaders
from multiprocessing import Pool
import pickle

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# # Goal
#
# * Filter data for omnipresent devices, which are a bad proxy for social interactions
# * Group data
#



def concatenater(args, frac=0.70):
    """
    Concatenates users in timebin, present `frac` part of the time
    """
    vc = args.value_counts()
    return set(vc[vc >= frac*vc.max()].index)


def mostly_present_counter(args):
    return len(concatenater(args))


def filterUserMac(*args):
    df, user, ua, evening_hour, morning_hour = args
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


def main(user):
    # ****************************************************************************
    # *                         Initial filtering of data                        *
    # ****************************************************************************
    try:
        print(f"Processing user {user}")
        ua = loaders.Useralias()  # noqa
        morning_hour = 7
        evening_hour = 17
        df = loaders.loadUserBluetooth(user, ua)
        if df is None:
            return None  # df is None because user have no bluetooth data
        cnt, var = filterUserMac(df, user, ua.userdct, evening_hour, morning_hour)
        remove_from_index = set(cnt[cnt > 30].index)
        df = df[~df.bt_mac.isin(remove_from_index)]

        # ****************************************************************************
        # *           Filter data to contain only free time before workdays          *
        # ****************************************************************************
        df['hour'] = df.index.hour  # noqa
        print("Done computing hour")
        df['weekday'] = df.index.weekday
        print("Done computing weekday")
        before_workday = df.weekday.isin({0, 1, 2, 3, 6})  # is it monday, tuesday, wendnesday, thursday or sunday?
        print("Done computing before_workday")
        free_time = (19 < df.hour) | (df.hour < 7)
        print("Done computing free_time")
        dfs = df[before_workday & free_time]

        dfs['scanned_user'] = dfs.scanned_user.replace(np.NaN, df.bt_mac)

        grouped = dfs.iloc[:3000].groupby('user')[['scanned_user']].resample('90T', closed='left').agg(concatenater)
        grouped['scanned_user'] = grouped.scanned_user.replace(set(), np.NaN)

        print("Fraction of non-nulls:", grouped.scanned_user.notnull().sum() / grouped.shape[0])
        print("Number of of non-nulls:", grouped.scanned_user.notnull().sum())
        return (grouped, var)
    except Exception as err:
        print(f"An Exception was raised when processing the user {user}:", file=sys.stderr)
        tb = sys.exc_info()[2]
        print(err.with_traceback(tb), file=sys.stderr)
        return None



if __name__ == '__main__':
    userlist = loaders.getUserList()
    try:
        pool = Pool(16)
        res = pool.map(main, userlist)
        grouped_res = {userlist[i]: res[i] for i in range(len(userlist))}
        with open('../../allan_data/binned_user_bluetooth_with_var.pkl', 'wb') as fid:
            pickle.dump(grouped_res, fid)
    except Exception as err:
        raise(err)
    finally:
        pool.close()
