#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

import numpy as np
from sklearn import model_selection
from sklearn import svm


n, p = 10000, 3
X = np.random.random((n, p))
y = np.random.random(n)
y = y > 0.3

param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]

svc = svm.SVC(kernel='linear')
gsv = model_selection.GridSearchCV(svc, param_grid)

gsv.fit(X, y)



# Apply the following patch to sklearn/model_selection/_search.py
"""
diff --git a/model_selection/__pycache__/_search.cpython-36.pyc b/model_selection/__pycache__/_search.cpython-36.pyc
index 3173cbf..d58eb43 100644
Binary files a/model_selection/__pycache__/_search.cpython-36.pyc and b/model_selection/__pycache__/_search.cpython-36.pyc differ
diff --git a/model_selection/_search.py b/model_selection/_search.py
index f574b39..cb85e00 100644
--- a/model_selection/_search.py
+++ b/model_selection/_search.py
@@ -625,6 +625,10 @@ class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
         base_estimator = clone(self.estimator)
         pre_dispatch = self.pre_dispatch

+        for parameters, (train, test) in product(candidate_params, cv.split(X, y, groups)):
+            trs, tes = train.size, test.size
+            print(trs/(trs+tes), tes/(trs+tes))
+
         out = Parallel(
             n_jobs=self.n_jobs, verbose=self.verbose,
             pre_dispatch=pre_dispatch
"""
