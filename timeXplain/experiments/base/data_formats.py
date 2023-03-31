import pandas as pd
from tslearn.utils import to_time_series_dataset


def encode_sktime_X(X):
    return pd.DataFrame({
        "dim_0": [pd.Series(_iloc(X)[x]) for x in range(len(X))]
    })


def encode_tslearn_X(X):
    return to_time_series_dataset(_values(X))


def encode_pyts_X(X):
    return _values(X)


def _values(A):
    if isinstance(A, pd.DataFrame):
        return A.values
    else:
        return A


def _iloc(A):
    if isinstance(A, pd.DataFrame):
        return A.iloc
    else:
        return A
