import sys
import os
sys.path.append(os.path.abspath(".."))
from speclib import pushbulletNotifier

import numpy as np
import pandas as pd
pd.set_option('display.width', 1200)  # for interactive use
import pickle

from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline

# from imblearn import over_sampling
# from imblearn import pipeline as imb_pipeline
# from imblearn import metrics as imb_metrics

# import warnings  # noqa
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)


# ****************************************************************************
# *                      Instantiate Pushbullet notifier                     *
# ****************************************************************************

pbn = pushbulletNotifier.JobNotification()


# ****************************************************************************
# *                       Settings for cross validation                      *
# ****************************************************************************

k_folds = 7
n_jobs = 40

cv_args = dict(scoring = 'roc_auc',        # noqa
               cv = k_folds,               # noqa
               verbose = 49,               # noqa
               refit = True,               # noqa
               n_jobs = n_jobs,            # noqa
               # pre_dispatch = 2 * n_jobs,  # noqa
               return_train_score = True)  # noqa

svc_kwargs = dict(probability = True,          # noqa
                  class_weight = 'balanced')  # noqa

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

try:
    svc_lin  = svm.SVC(kernel='linear', **svc_kwargs)
    pipe_lin = pipeline.Pipeline([('stsc', stsc), ('svc', svc_lin)])

    param_grid_lin = dict()
    param_grid_lin.update(svc_param_space_shared)

    est_lin      = model_selection.GridSearchCV(pipe_lin, param_grid_lin, **cv_args)
    est_lin.fit(X_tr, y_tr)
    _, prob1_lin = est_lin.best_estimator_.predict_proba(X_va).T
    AUC_lin      = metrics.roc_auc_score(y_va, prob1_lin)

    est_lin.best_auc_  = AUC_lin
    est_lin.X_tr       = X_tr
    est_lin.y_tr       = y_tr
    est_lin.X_va       = X_va
    est_lin.y_va       = y_va
    with open("gender_prediction_svm_lin_kernel_auc_score.pkl", "wb") as fid:
        pickle.dump(est_lin, fid)
except Exception as err:
    pbn.send(exception=err)
finally:
    msg = f"Completed cross validation with linear kernel, best AUC was {AUC_lin:.4f}"
    pbn.send(message=msg)


# ****************************************************************************
# *                              Polynomial SVC                              *
# ****************************************************************************

try:
    svc_poly  = svm.SVC(probability=True, kernel='poly')
    pipe_poly = pipeline.Pipeline([('stsc', stsc), ('svc', svc_poly)])

    param_grid_poly = {'svc__degree': [2, 3, 4, 5]}
    param_grid_poly.update(svc_param_space_shared)

    est_poly      = model_selection.GridSearchCV(pipe_poly, param_grid_poly, **cv_args)
    est_poly.fit(X_tr, y_tr)
    _, prob1_poly = est_poly.best_estimator_.predict_proba(X_va).T
    AUC_poly      = metrics.roc_auc_score(y_va, prob1_poly)

    est_poly.best_auc_ = AUC_poly
    est_poly.X_tr      = X_tr
    est_poly.y_tr      = y_tr
    est_poly.X_va      = X_va
    est_poly.y_va      = y_va
    with open("gender_prediction_svm_poly_kernel_auc_score.pkl", "wb") as fid:
        pickle.dump(est_poly, fid)
except Exception as err:
    pbn.send(exception=err)
finally:
    msg = f"Completed cross validation with polynomial kernel, best AUC was {AUC_poly:.4f}"
    pbn.send(message=msg)


# ****************************************************************************
# *                              Polynomial SVC                              *
# ****************************************************************************

try:
    svc_rbf  = svm.SVC(probability=True, kernel='poly')
    pipe_rbf = pipeline.Pipeline([('stsc', stsc), ('svc', svc_rbf)])

    param_grid_rbf = {'svc__gamma': 2.0**np.linspace(-10, 2, 13)}
    param_grid_rbf.update(svc_param_space_shared)

    est_rbf      = model_selection.GridSearchCV(pipe_rbf, param_grid_rbf, **cv_args)
    est_rbf.fit(X_tr, y_tr)
    _, prob1_rbf = est_rbf.best_estimator_.predict_proba(X_va).T
    AUC_rbf      = metrics.roc_auc_score(y_va, prob1_rbf)

    est_rbf.best_auc_  = AUC_rbf
    est_rbf.X_tr       = X_tr
    est_rbf.y_tr       = y_tr
    est_rbf.X_va       = X_va
    est_rbf.y_va       = y_va
    with open("gender_prediction_svm_rbf_kernel_auc_score.pkl", "wb") as fid:
        pickle.dump(est_rbf, fid)
except Exception as err:
    pbn.send(exception=err)
finally:
    msg = f"Completed cross validation with RBF kernel, best AUC was {AUC_rbf:.4f}"
    pbn.send(message=msg)
