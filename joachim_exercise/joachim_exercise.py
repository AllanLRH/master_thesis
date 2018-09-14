#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

from speclib import plotting, pushbulletNotifier

import numpy as np
import pandas as pd
import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, linear_model, metrics, model_selection
from sklearn.pipeline import Pipeline
import pickle
from time import sleep
import itertools
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(linewidth=145)

import logging
logging.basicConfig(filename='joachimExercise_expanded_feature_space_pca_reduction.log', level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('joachimExercise')

logger.info(f"Loading data")
df0 = pd.read_table('RGender.dat', sep=' ').T

df1 = df0.drop('gender', axis=1)
df1.rename(columns=lambda s: s.replace('bfi_', '').replace('.answer', ''), inplace=True)
gender = df0.gender
nd1 = preprocessing.scale(df1.values)

logger.info(f"Data loaded, expanding feature space")

# Expand the data with products accross _all_ variables
idx = np.arange(df1.values.shape[1]).astype(int)
idxCombinations = list(itertools.product(idx, idx))
featureMatrix = np.NaN * np.zeros((df1.values.shape[0], df1.values.shape[1]+len(idxCombinations)))

featureMatrix[:, :df1.values.shape[1]] = df1.values.copy()

for i, (v0, v1) in enumerate(idxCombinations, df1.values.shape[1]):
    featureMatrix[:, i] = featureMatrix[:, v0] * featureMatrix[:, v1]

logger.info(f"Expanded feature space, move on to pipeline operations")

jn = pushbulletNotifier.JobNotification(devices="phone")
jn.send(message="Started CV for joachim Exercise expanded feature space pca reduction coarse.")

processes = 23
try:
    x_re, x_va, y_re, y_va = model_selection.train_test_split(featureMatrix, gender.values,
                                                              test_size=0.2, stratify=gender.values)
    logger.info(f"Split data in to training set and validation set.")
    pipe = Pipeline([('pca', decomposition.PCA()),
                     ('lr', linear_model.LogisticRegression())])
    param_grid = {
        'pca__n_components': np.arange(20, 76),
        'lr__class_weight': ['balanced', None],
        'lr__C': 2.0**np.linspace(-9, 5, 20)
        }  # noqa
    logger.info(f"Starting cross validation")
    est = model_selection.GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=5, verbose=49, refit=True,
                                       n_jobs=processes, pre_dispatch=processes, return_train_score=True)
    est.fit(x_re, y_re)  # I think this is redundant
    # PCA transform the validation dataset with the number of components deemed optimal from the cross valiation
    x_va = pipe.named_steps['pca'].transform(x_va)
    _, yhat = est.predict_proba(x_va).T
    try:
        logger.info(f"Cross validation done, best score was {est.best_score_}")
        logger.info(f"Best params were {est.best_params_}")
        logger.info(f"Best estimator were {est.best_estimator_}")
        logger.info(f"Checking using the validation set.")
    except Exception as e:
        print(f"Logging exception: {e}")
    validation_auc_score = metrics.roc_auc_score(y_va, yhat)
    logger.info(f"AUC score for validation set of size {len(y_va)} is {validation_auc_score:.5f}")
    fig, ax, aucscore = plotting.plotROC(y_va, yhat)
    fig.savefig('figs/joachimExercise_expanded_feature_space_pca_reduction.pdf')
    est.y_va = y_va  # save for plotting ROC curve later
    est.yhat = yhat  # save for plotting ROC curve later
    est.validation_auc_score = validation_auc_score
    est.x_va = x_va
    est.y_va = y_va
    with open("joachimExercise_expanded_feature_space_pca_reduction.pkl", 'bw') as fid:
        pickle.dump(est, fid)
except Exception as err:
    jn.send(err)
    sleep(8)
    raise err

jn.send(message=f"Cross validation is done.")


