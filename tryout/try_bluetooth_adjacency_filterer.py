#!/usr/bin/env python
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np
np.random.seed(13)

users = "ABCDEF"
idx = pd.DatetimeIndex(freq='5T', start='2013-09-20 05:40', periods=10)
data = list()
for _ in idx:
    i, j = np.random.randint(0, len(users), size=(2,))
    data.append((set(users[i:j]), ))
df = pd.DataFrame(data, index=idx)
print(df, end='\n'*4)


def get_new_users_occuring_twice(df):
    # To keep track of users currently being observed.
    currently_observed = set()
    keep = dict()  # this is what's returned as a pd.Series
    for t0, t1 in zip(df.index[0:-1], df.index[1:]):
        observed_t0 = df.loc[t0].iloc[0]
        observed_t1 = df.loc[t1].iloc[0]
        # Intersection of observed users from t0 and t1
        user_intersect = observed_t0.intersection(observed_t1)
        # Remove the users allreaddy being observed from previous timesteps (like t-1, t-2, t-n...)
        new_users_occuring_twice = user_intersect - currently_observed
        if new_users_occuring_twice:  # if not empty
            keep[t0] = new_users_occuring_twice
        # Add the users which were just recorded to the set of users currently being ovserved...
        currently_observed.update(new_users_occuring_twice)
        # ... and those which weren't present in the t1-observeation (the most recent observation)
        currently_observed -= (currently_observed - observed_t1)
    return pd.Series(keep)

print(get_new_users_occuring_twice(df))
