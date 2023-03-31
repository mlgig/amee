from warnings import warn

import numpy as np
# We do not import the symbols in the following modules directly to reduce the risk of people accidentally using them
# (instead of the wrappers) or because collisions would occur.
import sklearn.tree
import sktime.classifiers.compose.ensemble as sktime_ens
import sktime.classifiers.distance_based.elastic_ensemble as sktime_ee
import sktime.classifiers.distance_based.proximity_forest as sktime_pf
import sktime.pipeline
import tslearn.svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sktime.transformers.compose import RowwiseTransformer, Tabulariser
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.shapelets import ContractedShapeletTransform
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf
from tslearn.metrics import cdist_gak

import lib.RotationForest.rotation_forest.rotation_forest as rf
import timexplain as tx
from experiments.base.data_formats import encode_sktime_X, encode_tslearn_X

RotationForestClassifier = rf.RotationForestClassifier


class _BaseWrapper(Pipeline):

    def __init__(self, estimator, transform_func):
        steps = [
            ("transform", FunctionTransformer(transform_func, validate=False, check_inverse=False)),
            ("estimator", estimator)
        ]
        super().__init__(steps)


class ElasticEnsembleClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(sktime_ee.ElasticEnsemble(**kwargs), encode_sktime_X)


class ProximityForestClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(sktime_pf.ProximityForest(**kwargs), encode_sktime_X)


class TimeSeriesForestClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(sktime_ens.TimeSeriesForestClassifier(**kwargs), encode_sktime_X)

    def predict_proba(self, X):
        # Workaround for this unexpected behavior: https://github.com/alan-turing-institute/sktime/issues/86
        probas = super().predict_proba(X)
        return np.expand_dims(probas, axis=0) if np.ndim(probas) == 1 else probas


# Taken from https://github.com/alan-turing-institute/sktime/blob/d302c26044785e4495b1d800c77da68920e8fca3/examples/time_series_classification.ipynb
class RISEClassifier(TimeSeriesForestClassifier):

    def __init__(self, **kwargs):
        base_estimator = sktime.pipeline.Pipeline([
            ("segment", RandomIntervalSegmenter(n_intervals=1, min_length=5)),
            ("features", sktime.pipeline.FeatureUnion([
                ("ar", RowwiseTransformer(
                    FunctionTransformer(func=RISEClassifier.ar_coefs, validate=False, check_inverse=False))),
                ("acf", RowwiseTransformer(
                    FunctionTransformer(func=RISEClassifier.acf_coefs, validate=False, check_inverse=False))),
                ("ps", RowwiseTransformer(
                    FunctionTransformer(func=RISEClassifier.powerspectrum, validate=False, check_inverse=False)))
            ])),
            ("tabularise", Tabulariser()),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)),
            # to fix NaNs that have sneaked in
            ("dt", sklearn.tree.DecisionTreeClassifier())
        ])
        super().__init__(base_estimator=base_estimator, **kwargs)

    @staticmethod
    def ar_coefs(x, maxlag=100):
        nlags = np.minimum(len(x) - 1, maxlag)
        model = AR(endog=x)
        return model.fit(maxlag=nlags, trend="nc").params

    @staticmethod
    def acf_coefs(x, maxlag=100):
        nlags = np.minimum(len(x) - 1, maxlag)
        return acf(x, nlags=nlags)

    @staticmethod
    def powerspectrum(x, **kwargs):
        fft = np.fft.fft(x)
        ps = fft.real * fft.real + fft.imag * fft.imag
        return ps[:ps.shape[0] // 2]


class ShapeletTransformClassifier(Pipeline):

    def __init__(self, **kwargs):
        steps = [
            ("transform", FunctionTransformer(encode_sktime_X, validate=False, check_inverse=False)),
            ("st", ContractedShapeletTransform(**kwargs)),
            ("rf", RandomForestClassifier(n_estimators=100))
        ]
        super().__init__(steps)

    # Post-hoc replacement of ShapeletTransform.transform() by our own, substantially faster implementation.
    def predict(self, X, **predict_params):
        return self["rf"].predict(self._transform_until_rf(X), **predict_params)

    def predict_proba(self, X):
        return self["rf"].predict_proba(self._transform_until_rf(X))

    def _transform_until_rf(self, X):
        return tx.spec.align_shapelets(self["st"], self["transform"].transform(X))[0]


class TimeSeriesSVC(_BaseWrapper):

    def __init__(self, kernel="linear", **kwargs):
        if kernel == "gak":
            warn("This GAK implementation performs bad on some data for an unknown reason. "
                 "We suggest you refrain from using it.", stacklevel=2)
            kernel = self._kernel_func_gak
        self.estimator = tslearn.svm.TimeSeriesSVC(kernel=kernel, **kwargs)
        super().__init__(self.estimator, encode_tslearn_X)

    # Hack which is required because pickle cannot serialize lambdas
    # and tslearn uses lambdas for its GAK kernel function.
    def _kernel_func_gak(self, x, y):
        g = self.estimator.gamma
        sz = self.estimator.sz
        d = self.estimator.d
        if g == "auto":
            g = 1.0
        return cdist_gak(x.reshape((-1, sz, d)), y.reshape((-1, sz, d)), sigma=np.sqrt(g / 2.0))
