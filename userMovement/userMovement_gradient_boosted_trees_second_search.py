import sys
import os
sys.path.append(os.path.abspath(".."))
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt  # noqa
import pickle
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

from speclib import pushbulletNotifier, plotting, misc  # noqa

np.set_printoptions(linewidth=145)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics, model_selection

import logging
logging.basicConfig(filename='userMovement_gradient_bboosting_classifier_second_search.log', level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('userMovement_gradient_bboosting_classifier')

data_path = '../../allan_data/DataPredictMovement_half.p'
x, y = np.load(data_path)
x = x.astype(float)
logger.info(f"Loaded data")


jn = pushbulletNotifier.JobNotification(devices="phone")

np.random.seed(476471)
processes = 28
try:
    x_re, x_va, y_re, y_va = model_selection.train_test_split(x, y, test_size=0.2, stratify=y)
    logger.info(f"Split data in to training set and validation set.")
    est0 = GradientBoostingClassifier(presort=True)
    param_grid = {
                        # (np.exp(np.log(156)/(15)) ** np.arange(15, 21, 0.5)).astype(int)
        'n_estimators': np.array([155, 184, 218, 258, 305, 361, 428, 506, 599, 709, 839, 993]),
        'learning_rate': np.logspace(-1, -0.01, 6, base=2),
        'max_depth': [5, 6, 7]
        }  # noqa
    logger.info(f"Starting cross validation")
    est = model_selection.GridSearchCV(est0, param_grid, scoring='roc_auc', cv=4, verbose=49, refit=True,
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
    fig.savefig('figs/userMovement_cv_GradientBoostingClassifier_second_search.pdf')
    est.y_va = y_va  # save for plotting ROC curve later
    est.yhat = yhat  # save for plotting ROC curve later
    est.validation_auc_score = validation_auc_score
    est.x_va = x_va
    est.y_va = y_va
    with open("userMovement_cv_GradientBoostingClassifier_second_search.pkl", 'bw') as fid:
        pickle.dump(est, fid)
except Exception as err:
    jn.send(err)
    sleep(8)
    raise err

jn.send(message=f"Cross validation is done.")
