#!/usr/bin/env python
# -*- coding: utf8 -*-


import sys
import os
sys.path.append(os.path.abspath(".."))
import traceback

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa
import seaborn as sns
import networkx as nx  # noqa
import re  # noqa
import itertools  # noqa
import pickle
from multiprocessing import Pool
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

from speclib import misc, loaders  # noqa
from speclib.pushbulletNotifier import JobNotification

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main(user, ua, phonebook):
    # ## Check bluetooth
    #
    userlist_2 = userlist[::]
    keep_users = list()
    for user in userlist_2[:35]:
        ub = loaders.loadUserBluetooth(user, ua)
        if ub is None:
            continue
        ub = ub.dropna()
        bt_200h = ub.index.max() - ub.index.min() > pd.Timedelta('200H')
        if bt_200h:
            keep_users.append(user)
    userlist_2 = keep_users


    # ## Check sms
    #
    # * At least 90 sms data for 90 days
    keep_users = list()
    for user in userlist_2:
        us = loaders.loadUser2(user, dataFilter=('sms',))
        us = pd.DataFrame(us['sms'])
        us = us.rename(columns={'address': 'number'})

        us['user'] = us.user.replace(ua.userdct, inplace=None)
        us['timestamp'] = pd.to_datetime(us.timestamp, unit='s', infer_datetime_format=True)
        us['number'] = us.number.replace(phonebook, inplace=None)
        us = us[us.number.str.len() == 5]

        sms_at_least_90_days = np.ptp(us.timestamp) > pd.Timedelta('90D')
        at_least_950_sms = us.shape[0] >= 950
        if (sms_at_least_90_days and at_least_950_sms):
            keep_users.append(user)
    userlist_2 = keep_users


    # ## Check calls
    #
    # * At least 90 call data for 90 days
    keep_users = list()
    for user in userlist_2:
        uc = loaders.loadUser2(user, dataFilter=('call',))
        uc = pd.DataFrame(uc['call'])

        uc['user'] = uc.user.replace(ua.userdct, inplace=None)
        uc['timestamp'] = pd.to_datetime(uc.timestamp, unit='s', infer_datetime_format=True)
        uc['number'] = uc.number.replace(phonebook, inplace=None)
        uc = uc[uc.number.str.len() == 5]
        call_at_least_90_days = np.ptp(uc.timestamp) > pd.Timedelta('90D')
        at_least_170_calls = uc.shape[0] >= 170
        if (call_at_least_90_days and at_least_170_calls):
            keep_users.append(user)
    return keep_users


if __name__ == '__main__':
    jn = JobNotification(devices="phone")
    try:
        userlist = loaders.getUserList()[:24]
        ua = loaders.Useralias()
        with open('/lscr_paper/allan/phonenumbers.p', 'rb') as fid:
            phonebook = pickle.load(fid)
        phonebook = {k: ua.userdct[v] for (k, v) in phonebook.items() if v in ua.userdct}

        with Pool(24) as pool:
            valid_users = pool.starmap(main, [(user, ua, phonebook) for user in userlist])

        print(f"Found {len(valid_users)} valid users")

        with open("valid_users.txt", 'w') as fid:
            fid.write('\n'.join(valid_users))
    except Exception as err:
        jn.send(err)
    jn.send()
