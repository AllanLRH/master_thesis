#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))


import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt  # noqa
from sklearn import preprocessing, linear_model, metrics, model_selection
from imblearn.pipeline import make_pipeline
import pickle
from time import sleep
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

from speclib import plotting, pushbulletNotifier
from imblearn import over_sampling

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

df_dummy_lst = [pd.get_dummies(df1[col], prefix=col) for col in df1.columns]
dfd = df_dummy_lst[0].copy()
df_dummy_lst = df_dummy_lst[1:]
dfd = dfd.join(df_dummy_lst)
del df_dummy_lst

logger.info(f"Data loaded")

jn = pushbulletNotifier.JobNotification(devices="phone")

processes = 28
try:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dfd.values, gender.values,
                                                                        test_size=0.2, stratify=gender.values)

    logger.info(f"Split data in to training set and validation set.")
    classifier = ['lr', linear_model.LogisticRegression(solver='liblinear', max_iter=1200)]
    sampler_lst = [['SMOTE', over_sampling.SMOTE()],
                   ['ADASYN', over_sampling.ADASYN()],
                   ['RandomOverSampler', over_sampling.RandomOverSampler()]]
    pipeline_lst = [ [f'{sampler[0]}-{classifier[0]}', make_pipeline(sampler[1], classifier[1])]
                      for sampler in sampler_lst ]  # noqa
    param_grid = {
        'lr__penalty': ['l1', 'l2'],
        'lr__C': 2.0**np.linspace(-6, 5, 15)
        }  # noqa
    for name, pipe in pipeline_lst:
        jn.send(message=f"Starding cross validation with resampling method {name}")
        logger.info(f"Starting cross validation")
        est = model_selection.GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=5, verbose=49, refit=True,
                                           n_jobs=processes, pre_dispatch=processes, return_train_score=True)
        # PCA transform the validation dataset with the number of components deemed optimal from the cross valiation
        _, yhat = est.best_estimator_.predict_proba(X_test).T
        try:
            logger.info(f"Cross validation done, best score was {est.best_score_}")
            logger.info(f"Best params were {est.best_params_}")
            logger.info(f"Best estimator were {est.best_estimator_}")
            logger.info(f"Checking using the validation set.")
        except Exception as e:
            print(f"Logging exception: {e}")
        validation_auc_score = metrics.roc_auc_score(y_test, yhat)
        logger.info(f"AUC score for validation set of size {len(y_test)} is {validation_auc_score:.5f}")
        logger.info(f"AUC score for validation set of size {len(y_test)} is {validation_auc_score:.5f}")
        fig, ax, aucscore = plotting.plotROC(y_test, yhat)
        fig.savefig(f'figs/joachim_exercise_resampling_binarized_sparse_{name}_resampled.pdf')
        est.y_test = y_test  # save for plotting ROC curve later
        est.yhat = yhat  # save for plotting ROC curve later
        est.validation_auc_score = validation_auc_score
        est.X_test = X_test
        est.y_test = y_test
        with open("joachim_exercise_resampling_binarized_sparse_{name}_resampled.pkl", 'bw') as fid:
            pickle.dump(est, fid)
except Exception as err:
    jn.send(err)
    sleep(8)
    raise err

jn.send(message=f"Cross validation is done.")
