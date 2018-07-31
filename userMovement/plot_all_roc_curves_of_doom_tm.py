#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
from glob import glob
sys.path.append(os.path.abspath(".."))
import matplotlib as mpl
mpl.use('pdf')
import pickle
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
mpl.rcParams['text.usetex'] = True

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from speclib import plotting, misc
from sklearn import model_selection
import pandas as pd

np.set_printoptions(linewidth=145)
pickles = glob("*.pkl")
# print(*pickles, sep='\n')

data_path = '../../allan_data/DataPredictMovement_half.p'
x, y = np.load(data_path)
del x
x = pd.read_msgpack('../../allan_data/DataPredictMovement_change_of_movement_indicator_state.msgpack').values
x_re, x_va, y_re, y_va = model_selection.train_test_split(x, y, test_size=0.2, stratify=y)


def lpk(pkl):
    with open(pkl, 'br') as fid:
        est = pickle.load(fid)
        print("Loaded", pkl)
    return est


statechange          = lpk('usermove_statechange.pkl')
sgd_std_final_coarse = lpk('userMovement_sgd_std_final_coarse.pkl')
cv_subgrid_search    = lpk('userMovement_cv_subgrid_search.pkl')
rf_coarse            = lpk('userMovement_rf_coarse.pkl')

statechange.best_estimator_.fit(x_re, y_re)
print('Predicted statechange')
statechange_prob          = statechange.predict_proba(x_va)[:, 1]
print('Predicted statechange_prob')
rf_coarse_prob            = rf_coarse.best_estimator_.predict_proba(rf_coarse.x_va)[:, 1]
print('Predicted rf_coarse_prob')
cv_subgrid_search_prob    = cv_subgrid_search.best_estimator_.predict_proba(cv_subgrid_search.x_va)[:, 1]
print('Predicted cv_subgrid_search_prob')
sgd_std_final_coarse_prob = sgd_std_final_coarse.best_estimator_.predict_proba(sgd_std_final_coarse.x_va)[:, 1]
print('Predicted sgd_std_final_coarse_prob')

fig, ax = plt.subplots()
plotting.plotROC(y_va, statechange_prob, ax=ax, label='statechange, Log', alpha=0.55)  # noqa
plotting.plotROC(rf_coarse.y_va, rf_coarse_prob, ax=ax, label='RF, coarse', alpha=0.55)  # noqa
plotting.plotROC(sgd_std_final_coarse.y_va, sgd_std_final_coarse_prob, ax=ax, label='SGD Log, coarse', alpha=0.55)  # noqa
plotting.plotROC(cv_subgrid_search.y_va, cv_subgrid_search_prob, ax=ax, label='SGD Log, fine', alpha=0.55)  # noqa
ax.legend(loc='lower right')
fig.savefig('figs/combined_auc.png', dpi=400)
