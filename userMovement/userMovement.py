import sys
import os
sys.path.append(os.path.abspath(".."))

from time import sleep
import numpy as np
# import bottleneck as bn
import pandas as pd
# import igraph as ig
import matplotlib as mpl
mpl.use('agg')
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
mpl.rcParams['text.usetex'] = False

# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)

import logging
logging.basicConfig(filename='userMovement.log', level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('userMovement')

from speclib import misc, loaders, plotting, modeleval, pushbulletNotifier

pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

jn = pushbulletNotifier.JobNotification(devices="phone")

try:
    data_path = '../../allan_data/DataPredictMovement_half.p'
    x, y = np.load(data_path)
    x = x.astype(float)
    logger.info(f"Loaded data")

    n1 = 800
    sample_idx1 = np.random.choice(range(len(y)), n1)
    x1 = x[sample_idx1, :]
    y1 = y[sample_idx1]
    logger.info(f"Downsampled data to {n1} samples")

    x_re, x_va, y_re, y_va = model_selection.train_test_split(x1, y1, test_size=0.2, stratify=y)
    logger.info(f"Split data in to training set and validation set.")
    pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('pca', PCA()), ('svm', SVC(probability=True))])
    param_grid = {
        'pca__n_components': np.hstack([np.arange(1, 9), np.arange(9, 33, 2)]),
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svm__degree': np.arange(1, 5),
        'svm__gamma': np.logspace(-6, 7, num=15),  # 10.0**np.arange(-8, 8),
        'svm__C': np.logspace(-6, 7, num=15)
        }  # noqa
    logger.info(f"Starting cross validation")
    est = model_selection.GridSearchCV(pipe, param_grid, scoring='roc_auc', n_jobs=42, cv=4, verbose=2)
    logger.info(f"Cross validation done.")
    est.fit(x_re, y_re)

except Exception as err:
    jn.send(err)
    sleep(8)
    raise err





jn.send(message="Cross validation is done")
