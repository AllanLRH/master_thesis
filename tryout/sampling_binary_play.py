#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

import numpy as np
from imblearn import over_sampling


class ADASYN_Categorical(over_sampling.ADASYN):
    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 n_neighbors=5,
                 n_jobs=1,
                 ratio=None):
        # super(over_sampling.ADASYN, self).__init__(sampling_strategy='auto')
        super(ADASYN_Categorical, self).__init__(sampling_strategy=sampling_strategy, ratio=ratio)

    def sample(self, X, y):
        res = super(ADASYN_Categorical, self).sample(X, y)
        res = np.round(res)
        return res







