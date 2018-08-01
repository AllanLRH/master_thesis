#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
colorcycle = [(0.498, 0.788, 0.498),
              (0.745, 0.682, 0.831),
              (0.992, 0.753, 0.525),
              (0.220, 0.424, 0.690),
              (0.749, 0.357, 0.090),
              (1.000, 1.000, 0.600),
              (0.941, 0.008, 0.498),
              (0.400, 0.400, 0.400)]


pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)

sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=2.8)
sns.set_palette(colorcycle)
mpl.rcParams['figure.max_open_warning'] = 65
mpl.rcParams['figure.figsize'] = np.array([12, 7]) * 0.75
mpl.rcParams['text.usetex'] = True
import json


with open('userMovement_sgd_std_final_coarse.pkl', 'br') as fid:
    est = pickle.load(fid)
cvr = pd.DataFrame(est.cv_results_)

description_file = "figs/crossval_pca_alpha_descriptions.json"
description_dict = dict()

for weight, penalty, loss in itertools.product(cvr.param_sgd__class_weight.unique(), cvr.param_sgd__penalty.unique(), cvr.param_sgd__loss.unique()):
    print(f'Processing combination {weight}, {penalty}, {loss}')
    if weight is None:
        mask = cvr.param_sgd__class_weight.isna() & (cvr.param_sgd__penalty == penalty) & (cvr.param_sgd__loss == loss)
    else:
        mask = (cvr.param_sgd__class_weight == weight) & (cvr.param_sgd__penalty == penalty) & (cvr.param_sgd__loss == loss)
    name = f'weight_{weight}_penalty_{penalty}_loss_{loss}'

    scores = cvr[mask][['mean_train_score', 'mean_test_score', 'std_train_score', 'std_test_score']]
    scores.rename(columns={'mean_train_score': 'Mean training score',
                           'mean_test_score': 'Mean testing score',
                           'std_test_score': 'Std testing score',
                           'std_train_score': 'Std training score'}, inplace=True)
    # description_dict[name] = scores.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_latex()
    description_dict[name] = scores.describe().to_latex()

    to_plot_mean = cvr[mask].groupby(['param_pca__n_components', 'param_sgd__alpha']).mean_test_score.mean().T.unstack()
    to_plot_std = cvr[mask].groupby(['param_pca__n_components', 'param_sgd__alpha']).std_test_score.mean().T.unstack()
    to_plot_mean.columns.name = 'alpha'
    to_plot_std.columns.name = 'alpha'
    to_plot_mean.index.name = 'PCA components'
    to_plot_std.index.name = 'PCA components'
    xticklabels = ["$10^{%.2f}$" % el for el in to_plot_mean.columns]
    xticklabels = ["$10^{%.2f}$" % el for el in to_plot_std.columns]
    yticklabels = ["${}$".format(el) for el in to_plot_mean.index]
    yticklabels = ["${}$".format(el) for el in to_plot_std.index]
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=[12, 7])
        sns.heatmap(to_plot_mean, ax=ax, cmap='inferno', vmin=0.74, vmax=0.86)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('PCA components')
        ax.set_xticklabels(xticklabels, rotation='vertical')
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        fig.savefig(f'figs/crossval_pca_alpha_rough_{name}_mean_score.pdf')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=[12, 7])
        sns.heatmap(to_plot_std, ax=ax, cmap='inferno', vmin=0.0, vmax=0.0015)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('PCA components')
        ax.set_xticklabels(xticklabels, rotation='vertical')
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        fig.savefig(f'figs/crossval_pca_alpha_rough_{name}_std_score.pdf')

with open(description_file, 'w') as fid:
    json.dump(description_dict, fid, indent=4)

