from itertools import product

import numpy as np
import pyts.classification
import pyts.transformation
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from tqdm.auto import tqdm

from experiments.base.data_formats import encode_pyts_X


class _BasePrunedGridSearchSoftVoting(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        always_iterable_values = [v if np.iterable(v) else [v] for v in kwargs.values()]
        self.param_choices = [dict(zip(kwargs.keys(), values)) for values in product(*always_iterable_values)]

    def fit(self, X, y, pbar=True):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        self.le_ = LabelEncoder().fit(y)
        transformed_y_train = self.le_.transform(y_train)
        transformed_y_val = self.le_.transform(y_val)

        clfs_accuracies = []
        for params in tqdm(self.param_choices, leave=False, disable=not pbar):
            clf = self._estimator_constr(**params)
            pred = clf.fit(X_train, transformed_y_train).predict(X_val)
            accuracy = accuracy_score(transformed_y_val, pred)
            clfs_accuracies.append((clf, accuracy))
        max_accuracy = max(acc for cld, acc in clfs_accuracies)
        self.estimators_ = [clf for clf, acc in clfs_accuracies if acc >= max_accuracy * 0.99]
        return self

    def _estimator_constr(self, **params):
        raise NotImplementedError

    def predict(self, X, pbar=True):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X, pbar), axis=1))

    def predict_proba(self, X, pbar=True):
        return np.average([clf.predict_proba(X) for clf
                           in tqdm(self.estimators_, leave=False, disable=not pbar)], axis=0)


class SAXVSMEnsembleClassifier(_BasePrunedGridSearchSoftVoting):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _estimator_constr(self, **params):
        return Pipeline([
            ("transform", FunctionTransformer(encode_pyts_X, validate=False, check_inverse=False)),
            ("sax_vsm", pyts.classification.SAXVSM(**params))
        ])

    def fit(self, X, y, pbar=True):
        valid = np.max(X, axis=1) - np.min(X, axis=1) != 0
        super().fit(X[valid], y[valid], pbar)

    def predict_proba(self, X, pbar=True):
        valid = np.max(X, axis=1) - np.min(X, axis=1) != 0
        n_classes = len(self.estimators_[0]["sax_vsm"].classes_)
        # All invalid samples get the same probability for each class.
        proba = np.full((X.shape[0], n_classes), 1 / n_classes)
        if np.any(valid):
            proba[valid] = np.average([softmax(clf.decision_function(X[valid]), axis=-1)
                                       for clf in tqdm(self.estimators_, leave=False, disable=not pbar)], axis=0)
        return proba


class BOSSEnsembleClassifier(_BasePrunedGridSearchSoftVoting):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _estimator_constr(self, **params):
        def _toarray(x):
            return x.toarray()

        return Pipeline([
            ("to_pyts", FunctionTransformer(encode_pyts_X, validate=False, check_inverse=False)),
            ("boss", pyts.transformation.BOSS(**params)),
            ("to_dense", FunctionTransformer(_toarray, validate=False, check_inverse=False)),
            ("knn", pyts.classification.KNeighborsClassifier(n_neighbors=1, metric="boss"))
        ])


class WEASELEnsembleClassifier(_BasePrunedGridSearchSoftVoting):

    def __init__(self, window_sizes=np.arange(0.1, 1, 0.01), window_steps=None, **kwargs):
        self.window_sizes = window_sizes
        self.window_steps = window_steps
        super().__init__(**kwargs)

    def _estimator_constr(self, **params):
        return Pipeline([
            ("to_pyts", FunctionTransformer(encode_pyts_X, validate=False, check_inverse=False)),
            ("weasel",
             pyts.transformation.WEASEL(window_sizes=self.window_sizes, window_steps=self.window_steps, **params)),
            ("logreg", LogisticRegression(penalty="l2", C=1, fit_intercept=True, solver="liblinear", multi_class="ovr"))
        ])
