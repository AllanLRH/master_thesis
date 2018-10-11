#!/usr/bin/env pythonw
# -*- coding: utf8 -*-
import sys
import os
sys.path.append(os.path.abspath(".."))
import matplotlib as mpl
mpl.use('agg')
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
import pandas as pd
from time import sleep

from imblearn.pipeline import make_pipeline
from imblearn import over_sampling

from speclib import pushbulletNotifier, plotting, misc

np.set_printoptions(linewidth=145)

from sklearn import ensemble
# from sklearn.svm import LinearSVC, SVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, model_selection, preprocessing
from sklearn.pipeline import Pipeline

# # ****************************************************************************
# # *  Will wait for CPU resources to be idle before executing past this point *
# # ****************************************************************************
# misc.wait_for_cpu_resources()

import logging
from imp import reload
reload(logging)
logging.basicConfig(filename='userMovement_rf_oversampled.log', level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('userMovement_rf_oversampled')

data_path = '../../allan_data/DataPredictMovement_half.p'
x, y = np.load(data_path)
x = x.astype(float)
logger.info(f"Loaded data")


jn = pushbulletNotifier.JobNotification(devices="phone")
jn.send(message="Started CV for RF grid with oversampling.")

processes = 12
try:
    x_re, x_va, y_re, y_va = model_selection.train_test_split(x, y, test_size=0.2, stratify=y)
    logger.info(f"Split data in to training set and validation set.")

    # BUG: Must be called 'randomforestclassifier', since that seems to be used for the internal representation
    # no matter the name given for the classifier.
    classifier = ['randomforestclassifier', ensemble.RandomForestClassifier()]
    sampler_lst = [['smote', over_sampling.SMOTE()],
                   ['adasyn', over_sampling.ADASYN()],
                   ['random¬oversampler', over_sampling.RandomOverSampler()]]
    pipeline_lst = [ [f'{sampler[0]}-{classifier[0]}', make_pipeline(sampler[1], classifier[1])]
                      for sampler in sampler_lst ]  # noqa
    param_grid = {
                  'randomforestclassifier__n_estimators': np.arange(40, 100, 10),
                  'randomforestclassifier__max_depth': np.arange(13, 20)
                 }

    for name, pipe in pipeline_lst:
        jn.send(message=f"Starding cross validation with resampling method {name}")
        logger.info(f"Starding cross validation with resampling method {name}")
        est = model_selection.GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=5, verbose=49, refit=True,
                                           n_jobs=processes, pre_dispatch=processes, return_train_score=True)
        est.fit(x_re, y_re)  # I think this is redundant
        _, yhat = est.predict_proba(x_va).T
        try:
            logger.info(f"Cross validation done, best score was {est.best_score_} using {name} oversampling.")
            logger.info(f"Best params were {est.best_params_} using {name} oversampling.")
            logger.info(f"Best estimator were {est.best_estimator_} using {name} oversampling.")
            logger.info(f"Checking using the validation set.")
        except Exception as e:
            print(f"Logging exception: {e}")
        validation_auc_score = metrics.roc_auc_score(y_va, yhat)
        logger.info(f"AUC score for validation set of size {len(y_va)} using {name} sampling is {validation_auc_score:.5f}")
        fig, ax, aucscore = plotting.plotROC(y_va, yhat)
        fig.savefig(f'figs/userMovement_rf_{name}_oversampled.pdf')
        est.y_va = y_va  # save for plotting ROC curve later
        est.yhat = yhat  # save for plotting ROC curve later
        est.validation_auc_score = validation_auc_score
        est.x_va = x_va
        est.y_va = y_va
        with open(f"userMovement_rf_{name}_oversampled.pkl", 'bw') as fid:
            pickle.dump(est, fid)
except Exception as err:
    jn.send(err)
    sleep(8)
    raise err

jn.send(message=f"Cross validation is done.")

