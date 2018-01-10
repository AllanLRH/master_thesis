#!/usr/bin/env python
# -*- coding: utf8 -*-

import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn import metrics
from sklearn import model_selection


def stratifiedCrossEval(X, y, model, metricFunctions=None, n_splits=5, test_size=0.3):
    """Do a stratified Cross evaluation of a model to gauge it's performance.

    Parameters
    ----------
    X : np.ndarray
        Feature space.
    y : np.ndarray
        Correct prediction.
    model : Scikit Learn Classifier
        Predict_proba will be used for model evaluation if it's avaiable.
    metricFunctions : list of (str, function), optional
        List with [(functionname, function)]. Function must take the arguments
        (true_prediction, predicted).
        Default is AUC ad accuracy.
    n_splits : int, optional
        Number of splits to use in the cross evaluation.
        Default is 5.
    test_size : float, optional
        Fraction of dataset used for the test data.
        Default is 0.3

    Returns
    -------
    pd.DataFrame
        DataFrame with the performance of all metrics for each evaluation.
    """
    if metricFunctions is None:
        metricFunctions = [('AUC', metrics.roc_auc_score),
                           ('accuracy', lambda tru, pre: metrics.accuracy_score(tru, pre > 0.5))]
    metricNames, _ = tuple(zip(*metricFunctions))  # unpack metricFunctions

    sss = model_selection.StratifiedShuffleSplit(n_splits, test_size)

    # for storing results of metric evaluation
    df = pd.DataFrame(index=np.arange(n_splits), columns=metricNames)

    # Do the cross validation
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]

        # sanity check
        assert X_tr.shape[0] == y_tr.shape[0], f"Dimmension mismatch X_tr.shape: {X_tr.shape} y_tr.shape: {y_tr.shape}"
        assert X_te.shape[0] == y_te.shape[0], f"Dimmension mismatch X_te.shape: {X_te.shape} y_te.shape: {y_te.shape}"

        model.fit(X_tr, y_tr)
        if hasattr(model, 'predict_proba'):
            model_prediction = model.predict_proba(X_te)[:, 1]
        else:
            model_prediction = model.predict(X_te)[:, 1]


        # Evalueate metrics
        for name, func in metricFunctions:
            df.loc[i, name] = func(y_te, model_prediction)
    return df
