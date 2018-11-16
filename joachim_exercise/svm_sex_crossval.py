import sys  # noqa
import os  # noqa
sys.path.append(os.path.abspath(".."))
from speclib import misc  # noqa
from speclib import loaders  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa
pd.set_option('display.width', 1200)  # for interactive use
import pickle  # noqa

from sklearn import svm  # noqa
from sklearn import metrics  # noqa
from sklearn import decomposition  # noqa
from sklearn import model_selection  # noqa
from sklearn import preprocessing  # noqa
from sklearn import pipeline  # noqa
from imblearn import over_sampling  # noqa
from imblearn import pipeline as imb_pipeline  # noqa
from imblearn import metrics as imb_metrics  # noqa

# import warnings  # noqa
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)
from time import time


# ****************************************************************************
# *                       Settings for cross validation                      *
# ****************************************************************************

k_folds = 3
n_jobs = 40

cv_args = dict(scoring = 'roc_auc',        # noqa
               cv = k_folds,               # noqa
               verbose = 49,               # noqa
               refit = True,               # noqa
               n_jobs = n_jobs,            # noqa
               # pre_dispatch = 2 * n_jobs,  # noqa
               return_train_score = True)  # noqa

svc_kwargs = dict(probability = True,          # noqa
                  class_weights = 'balanced')  # noqa

svc_param_space_shared = {'svc__C': 2.0**np.linspace(-10, 10, 11)}

# ****************************************************************************
# *                                 Load data                                *
# ****************************************************************************

df = pd.read_table('RGender.dat', sep=' ').T
X = df.drop('gender', axis=1).values.astype(float)  # Convert to float to avoid warning during standardization
y = df.gender

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(X, y, test_size=0.2, stratify=y)

stsc = preprocessing.StandardScaler()


# ****************************************************************************
# *                                Linear SVC                                *
# ****************************************************************************

svc_lin  = svm.SVC(kernel='linear', **svc_kwargs)
pipe_lin = pipeline.Pipeline([('stsc', stsc), ('svc', svc_lin)])

param_grid_lin = dict()
param_grid_lin.update(svc_param_space_shared)

est_lin      = model_selection.GridSearchCV(pipe_lin, param_grid_lin, **cv_args)
est_lin.fit(X_tr, y_tr)
_, prob1_lin = est_lin.best_estimator_.predict_proba(X_va).T
AUC_lin      = metrics.roc_auc_score(y_va, prob1_lin)


# ****************************************************************************
# *                              Polynomial SVC                              *
# ****************************************************************************

svc_poly  = svm.SVC(probability=True, kernel='poly')
pipe_poly = pipeline.Pipeline([('stsc', stsc), ('svc', svc_poly)])

param_grid_poly = {'svc__degree': [2, 3, 4, 5]}
param_grid_poly.update(svc_param_space_shared)

est_poly      = model_selection.GridSearchCV(pipe_poly, param_grid_poly, **cv_args)
est_poly.fit(X_tr, y_tr)
_, prob1_poly = est_poly.best_estimator_.predict_proba(X_va).T
AUC_poly      = metrics.roc_auc_score(y_va, prob1_poly)


# ****************************************************************************
# *                              Polynomial SVC                              *
# ****************************************************************************

svc_rbf  = svm.SVC(probability=True, kernel='poly')
pipe_rbf = pipeline.Pipeline([('stsc', stsc), ('svc', svc_rbf)])

param_grid_rbf = {'svc__gamma': 2.0**np.linspace(-10, 2, 13)}
param_grid_rbf.update(svc_param_space_shared)

est_rbf      = model_selection.GridSearchCV(pipe_rbf, param_grid_rbf, **cv_args)
est_rbf.fit(X_tr, y_tr)
_, prob1_rbf = est_rbf.best_estimator_.predict_proba(X_va).T
AUC_rbf      = metrics.roc_auc_score(y_va, prob1_rbf)

