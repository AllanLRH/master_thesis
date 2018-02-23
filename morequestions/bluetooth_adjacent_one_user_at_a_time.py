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

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

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
    """
    Count interactions with all avaiable users.
    var is how the number of datapoints vary as users are removed (most present removed first.)
    """
    df, user, ua = args
    if df is None:
        return None
    # Group to day of year
    day_of_year_grouped = df.groupby(df.index.dayofyear)['bt_mac'].unique()
    # Count occurence of bluetooth mac addresses, summed for days
    cnt = pd.value_counts([el for el in itertools.chain(*day_of_year_grouped)])
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
        evening_hour = 18
        df = loaders.loadUserBluetooth(user, ua)
        if df is None:
            return None  # df is None because user have no bluetooth data
        cnt, var = filterUserMac(df, user, ua.userdct)
        remove_from_index = set(cnt[cnt > 30].index)
        df = df[~df.bt_mac.isin(remove_from_index)]

        # ****************************************************************************
        # *           Filter data to contain only free time before workdays          *
        # ****************************************************************************
        before_workday = df.index.weekday.isin({0, 1, 2, 3, 6})  # is it monday, tuesday, wendnesday, thursday or sunday?
        # print("Done computing before_workday")
        free_time = (evening_hour < df.index.hour) | (df.index.hour < morning_hour)
        # print("Done computing free_time")
        dfs = df[before_workday & free_time]

        dfs['scanned_user'] = dfs.scanned_user.replace(np.NaN, df.bt_mac)

        grouped = dfs.groupby('user')[['scanned_user']].resample('90T', closed='left').agg(concatenater)
        grouped['scanned_user'] = grouped.scanned_user.replace(set(), np.NaN)

        print(user, "fraction of non-nulls:", grouped.scanned_user.notnull().sum() / grouped.shape[0])
        print(user, "number of of non-nulls:", grouped.scanned_user.notnull().sum())
        return (grouped, var)
    except Exception as err:
        print(f"An Exception was raised when processing the user {user}:", file=sys.stderr)
        tb = sys.exc_info()[2]
        print(err.with_traceback(tb), file=sys.stderr)
        return None



if __name__ == '__main__':
    userlist = loaders.getUserList()
    ua = loaders.Useralias()
    try:
        pool        = Pool(22)
        res         = pool.map(main, userlist)
        grouped_res = {userlist[i]: res[i] for i in range(len(userlist)) if res[i] is not None}
        grouped_lst = list()
        var_lst     = list()
        for k, v in grouped_res.items():
            grouped_lst.append(v[0])
            var_df         = pd.DataFrame(v[1], columns=['count'])
            var_df['user'] = ua[k]
            var_df         = var_df.set_index(['user', var_df.index])
            var_lst.append(v[1])
        grouped_df                 = pd.concat(grouped_lst)
        grouped_df['scanned_user'] = grouped_df.scanned_user.map(list, na_action='ignore')
        var_df                     = pd.concat(var_lst)
        grouped_df.to_msgpack('../../allan_data/binned_user_bluetooth_grouped.msgpack')
        var_df.to_msgpack('../../allan_data/binned_user_bluetooth_var.msgpack')

    except Exception as err:
        raise(err)
    finally:
        pool.close()
