import sys
import os
sys.path.append(os.path.abspath(".."))
import matplotlib as mpl
mpl.use('agg')
import numpy as np
import panas as pd
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
from time import sleep

from speclib import pushbulletNotifier, plotting, misc

np.set_printoptions(linewidth=145)

from sklearn import metrics, model_selection, preprocessing, decomposition, linear_model
from sklearn.pipeline import Pipeline

# ****************************************************************************
# *  Will wait for CPU resources to be idle before executing past this point *
# ****************************************************************************
# misc.wait_for_cpu_resources()

import logging
logging.basicConfig(filename='kernel_pca_lr_predict_sex.log', level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('kernel_pca_lr_predict_sex')

df0 = pd.read_table('RGender.dat', sep=' ').T
df1 = df0.drop('gender', axis=1)
df1.rename(columns=lambda s: s.replace('bfi_', '').replace('.answer', ''), inplace=True)
gender = df0.gender
nd1 = preprocessing.scale(df1.values)
logger.info(f"Loaded data")


jn = pushbulletNotifier.JobNotification(devices="phone")
jn.send(message="Started CV for fine RF grid")

processes = 12
try:
    x_re, x_va, y_re, y_va = model_selection.train_test_split(nd1.values, gender, test_size=0.2, stratify=gender)
    logger.info(f"Split data in to training set and validation set.")
    pipe = Pipeline([('kpca', decomposition.KernelPCA(remove_zero_eig=True)),
                     ('lr', linear_model.LogisticRegression)])
    param_grid = {
        'kpca__kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
        'kpca__gamma': 1/np.arange(20, 80, 8),
        'kpca__degree': [2, 3, 4, 5],
        'kpca__n_components': x_re.shape[0]//np.arange(2, 6),
        'lr__C': 2.0**np.arange(-4, 4),
        'lr__class_weight': ['balanced', None]
        }  # noqa
    logger.info(f"Starting cross validation")
    est = model_selection.GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=5, verbose=49, refit=True,
                                       n_jobs=processes, pre_dispatch=processes, return_train_score=True)
    est.fit(x_re, y_re)  # I think this is redundant
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
    fig.savefig('figs/kernel_pca_lr_predict_sex_roc_curve.pdf')
    est.y_va = y_va  # save for plotting ROC curve later
    est.yhat = yhat  # save for plotting ROC curve later
    est.validation_auc_score = validation_auc_score
    est.x_va = x_va
    est.y_va = y_va
    with open('kernel_pca_lr_predict_sex_roc_curve.pkl', 'bw') as fid:
        pickle.dump(est, fid)
except Exception as err:
    jn.send(err)
    sleep(8)
    raise err

jn.send(message=f"Cross validation is done.")
