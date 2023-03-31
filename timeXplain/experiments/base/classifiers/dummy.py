from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import logistic
from sklearn.base import BaseEstimator, ClassifierMixin

from timexplain._utils import rfft_magnitudes


class _BaseThresholdClassifier(ABC, BaseEstimator, ClassifierMixin):
    def __init__(self, threshold, scale=1):
        self.threshold = threshold
        self.scale = scale

    @abstractmethod
    def _linear_measure(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        lin = self._linear_measure(X)
        proba = logistic.cdf(lin, loc=self.threshold, scale=self.scale)
        return np.stack([1 - proba, proba], axis=-1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)


class _BaseSliceMeanThresholdClassifier(_BaseThresholdClassifier):
    def __init__(self, slices, threshold, scale=1):
        super().__init__(threshold, scale)
        self.slices = slices

    @abstractmethod
    def _time_series_representation(self, X):
        raise NotImplementedError

    def _linear_measure(self, X):
        tsr = np.asarray(self._time_series_representation(X))
        return np.mean([np.mean(tsr[..., slc[0]:slc[1]], axis=np.ndim(X) - 1) for slc in self.slices], axis=0)


class DummyTimeSliceMeanThresholdClassifier(_BaseSliceMeanThresholdClassifier):
    def _time_series_representation(self, X):
        return X


class DummyFreqSliceMeanThresholdClassifier(_BaseSliceMeanThresholdClassifier):
    def _time_series_representation(self, X):
        return rfft_magnitudes(X)


class DummyVarianceThresholdClassifier(_BaseThresholdClassifier):
    def _linear_measure(self, X):
        return np.var(X, axis=np.ndim(X) - 1)
