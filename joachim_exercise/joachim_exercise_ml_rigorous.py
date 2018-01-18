#!/usr/bin/env python
# -*- coding: utf8 -*-

# In[1]:

import sys
import os
sys.path.append(os.path.abspath(".."))

from speclib import loaders
from speclib import plotting
from speclib import graph
from speclib import misc
from speclib import userActivityFunctions
from speclib import modeleval

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, linear_model, discriminant_analysis, neighbors
from sklearn import metrics, model_selection, svm, ensemble
import itertools
# from statMlFunctions import *
#import seaborn as sns
#sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=1.8)
#sns.set_palette(sns.color_palette("Set1", n_colors=12, desat=.5))
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

# sns.set(style="white")
# mpl.rcParams['figure.figsize'] = [10, 6]

mpl.rcParams['figure.max_open_warning'] = 65

# %load_ext watermark
# %watermark -a "Allan Leander Rostock Hansen" -u -d -v -p numpy,bottleneck,pandas,matplotlib,sklearn,missingno
# %watermark  -p networkx,igraph,seaborn,palettable


# In[2]:

df0 = pd.read_table('RGender.dat', sep=' ').T
df1 = df0.drop('gender', axis=1)
df1.rename(columns=lambda s: s.replace('bfi_', '').replace('.answer', ''), inplace=True)
df1.head()
gender = df0.gender
nd1 = preprocessing.scale(df1.values)


# In[3]:

pca = decomposition.PCA(svd_solver='full')
pca.fit(nd1)
td1 = pca.transform(nd1)  # The rotated vector space


# In[14]:

n_splits = 100
test_size = 0.3
subgrid_points = 20
score = 'accuracy'


# # Raw data

# ## Logistic regression

# In[18]:

tuned_parameters = [{'C': 10.0**np.arange(-3, 4), 'penalty': ['l1', 'l2']}]
print(tuned_parameters)


# In[19]:

perf_df, perf_df_sub, perf_eval_df = modeleval.fitAndEvalueateModel(nd1, gender.values,
                                                                    linear_model.LogisticRegression(),
                                                                    tuned_parameters, score, n_splits,
                                                                    test_size, subgrid_points)


# In[20]:

modeleval.summarizePerformanveEval(perf_eval_df, print_only=True)


# ## SVM

# In[ ]:

tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'],
                     'gamma': np.array([1e-2, 1e-2, 1e-3, 1e-4, 1e-5]),
                     'C': np.array([0.01, 0.1, 1, 10, 100, 1000])}]
print(tuned_parameters)


# In[ ]:

model = svm.SVC()
X = nd1
y = gender.values

perf_df = modeleval.gridsearchCrossVal(X, y, model, tuned_parameters, score)
best_params = dict(perf_df.loc[perf_df.accuracy.argmax(), 'best_params'])
subsearch_params = modeleval.construct_subsearch_tuned_parameters(best_params,
                                                                  tuned_parameters,
                                                                  n_gridpoints=n_gridpoints)
perf_df_sub = modeleval.gridsearchCrossVal(X, y, model, subsearch_params, score)
best_params_sub = dict(perf_df.loc[perf_df.accuracy.argmax(), 'best_params'])
perf_eval_df = modeleval.stratifiedCrossEval(X, y, model.set_params(**best_params_sub),
                                        n_splits=n_splits, test_size=test_size)


# In[ ]:

perf_df, perf_df_sub, perf_eval_df = modeleval.fitAndEvalueateModel(nd1, gender.values,
                                                                    svm.SVC(),
                                                                    tuned_parameters, score, n_splits,
                                                                    test_size, subgrid_points)


# In[22]:

svc = svm.SVC()


# In[25]:

svc.get_params().items()


# In[ ]:



