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
    df.index.name = 'fold'

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


def gridsearchCrossVal(X, y, model, tuned_parameters, score, n_jobs=75, n_splits=5, test_size=0.3):
    """Fit a model for the best parameters using a grid search with stratified cross valication.

    Parameters
    ----------
    X : np.ndarray
        Feature space, must be of size (n,) or (n, m).
    y : np.ndarray
        Target variable, must be of size (n, ).
    model : Scikit Learn model (an instance, not class definition)
        The model to be used in the classification.
    tuned_parameters : list with dict(s)
        Parameters to be used in the grid search
        List with dict(s), where the keys in a dict corresponds to the keyword arguments
        for the model. The keys maps to in iterable containing the values used in the
        grid search, and if the values are numeric, they should be a np.ndarray in order
        to be useful in conjunction with the function constructSubsearchTunedParameters.
    score : str
        The score to be used. Valid options are:
        accuracy, adjusted_mutual_info_score, adjusted_rand_score,
        average_precision, completeness_score, explained_variance,
        f1, f1_macro, f1_micro, f1_samples, f1_weighted,
        fowlkes_mallows_score, homogeneity_score, mutual_info_score,
        neg_log_loss, neg_mean_absolute_error,
        neg_mean_squared_error, neg_mean_squared_log_error,
        neg_median_absolute_error, normalized_mutual_info_score,
        precision, precision_macro, precision_micro,
        precision_samples, precision_weighted, r2, recall,
        recall_macro, recall_micro, recall_samples, recall_weighted,
        roc_auc, v_measure_score
    n_jobs : int, optional
        Number of cores to use in the grid search. Default is 75.
    n_splits : int, optional
        Number of splits/runs to do in the cross validation process. Default 5.
    test_size : float, optional
        Fraction of data to be reserved for the test data set. Default is 0.3.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with the results of the cross validation.
        The first column are the results of the score evaluation, the second column are
        a tuple with the arguments. NOTE that it's a tuple because a dict couldn't be
        embedded in the DataFrame, but it can (and should) be converted to a dict if it's
        to be used in a keywords arguments expansion.
    """
    sss = model_selection.StratifiedShuffleSplit(n_splits, test_size)
    clf = model_selection.GridSearchCV(model, param_grid=tuned_parameters, scoring=score, n_jobs=75)

    df = pd.DataFrame(index=np.arange(n_splits), columns=[score, 'best_params'])
    df.index.name = 'fold'

    # Do the cross validation
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]

        # sanity check
        assert X_tr.shape[0] == y_tr.shape[0], f"Dimmension mismatch X_tr.shape: {X_tr.shape} y_tr.shape: {y_tr.shape}"
        assert X_te.shape[0] == y_te.shape[0], f"Dimmension mismatch X_te.shape: {X_te.shape} y_te.shape: {y_te.shape}"

        # Do grid search
        clf.fit(X_tr, y_tr)

        # if hasattr(clf, 'predict_proba'):
        #     model_prediction = clf.predict_proba(X_te)
        # else:
        #     model_prediction = clf.predict(X_te)
        df.loc[i, score] = clf.best_score_
        df.loc[i, 'best_params'] = tuple(clf.best_params_.items())

    return df


def constructSubsearchTunedParameters(best_params, tuned_parameters_old, n_gridpoints=15):
    """Construct a new set of grid points for a sub-grid cross validation, given the old
       grid points and the best result of the previous cross valudation.
       It assunes that the ratio between point n-1 and n, and n and n+1 are the same, and
       will use "linspace((n-1), n+1, n_gridpoints)"" to search around the best_params
       value with index n.

    Parameters
    ----------
    best_params : dict
        **kwargs-like dict with parameters for the Scikit Learn model.
    tuned_parameters_old : list with dicts.
        The old set of search parameters, see the docs for gridsearchCrossVal.
    n_gridpoints : int, optional
        Number of points in the subgrid search. Default is 20.

    Returns
    -------
    list of dict(s)
        A net set of set of parameters to be tuned.

    Raises
    ------
    ValueError
        If the length if a numerical array in the original search space are less than 4.
    """
    newTuned = list()
    for dct in tuned_parameters_old:
        nDct = dict()
        for key, arr in dct.items():
            best = best_params[key]
            if isinstance(arr, np.ndarray):
                if arr.size <= 3:
                    raise ValueError("Array assiciated with key the {key} are of length {arr.size}, but need to be longer than 3.")
                idx = np.where(np.isclose(arr, best))[0][0]
                mid_idx = len(arr)//2
                point_ratio = arr[mid_idx]/arr[mid_idx+1]
                if point_ratio > 1:
                    point_ratio = 1/point_ratio
                nDct[key] = np.linspace(arr[idx]*point_ratio, arr[idx]/point_ratio, n_gridpoints)
            else:
                nDct[key] = arr
        newTuned.append(nDct)
    return newTuned


def fitAndEvalueateModel(X, y, model, tuned_parameters, score, n_splits=5, test_size=0.2,
                         n_gridpoints=20):
    """Wraps the functions gridsearchCrossVal, constructSubsearchTunedParameters and
    stratifiedCrossEval to easily
      1.  Do a coarse grid search with cross validation for fitting the model.
      2.  Construct a finer grid around the best parameters found in the previous step.
      3.  Repeat the grid search cross validation on the finer grid.
      4.  Evalueate the performance of the model using the new parameters.

    Parameters
    ----------
      See functions above.

    Returns
    -------
    tuple of (DataFrame, DataFrame, DataFrame)
        (DataFrame from coarse grid search, DataFrame from fine grid search,
         DataFrame from cross validation evaluation of model performance)
    """
    perf_df = gridsearchCrossVal(X, y, model, tuned_parameters, score)
    best_params = dict(perf_df.loc[perf_df.accuracy.values.argmax(), 'best_params'])
    subsearch_params = constructSubsearchTunedParameters(best_params, tuned_parameters,
                                                         n_gridpoints=n_gridpoints)
    perf_df_sub = gridsearchCrossVal(X, y, model, subsearch_params, score)
    best_params_sub = dict(perf_df.loc[perf_df.accuracy.values.argmax(), 'best_params'])
    perf_eval_df = stratifiedCrossEval(X, y, model.set_params(**best_params_sub),
                                       n_splits=n_splits, test_size=test_size)
    return (perf_df, perf_df_sub, perf_eval_df)


def summarizePerformanveEval(df, print_only=False):
    """Pretty-print the mean ± std of the dataframe output of stratifiedCrossEval.

    Parameters
    ----------
    df : Dataframe
        DataFrame from the stratifiedCrossEval function.
    print_only : bool, optional
        Don't return a list of strings, just print them. Default is False.

    Returns
    -------
    list or None
        List of pretty formatted strings, unless print_only is True.
    """
    mean = df.mean()
    std = df.std()
    str_lst = ["{} = {:0.3f} ± {:0.3f}".format(lbl, mean.loc[lbl], std.loc[lbl]) for lbl in mean.index]
    if print_only:
        print(*str_lst, sep='\n')
        return None
    return str_lst
