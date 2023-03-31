import os
import re
from glob import glob

import joblib
import keras
import numpy as np
import pandas as pd

from experiments.base.classifiers import KerasWrapper

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data"))
PREFITTED_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../prefitted"))

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../results"))
TEST_PREDS_DIR = os.path.join(RESULTS_DIR, "test_preds")
IMPACTS_DIR = os.path.join(RESULTS_DIR, "impacts")
EVALUATION_DATA_DIR = os.path.join(RESULTS_DIR, "evaluation_data")


def _get_path(path_template, *args):
    for arg in args:
        # Stop if we encounter a None which marks that there will be no more args.
        if arg is None:
            break

        open_pos = path_template.find("{")
        close_pos = path_template.find("}")

        value = str(arg)
        if close_pos == open_pos + 1:
            repl = value
        else:
            notempty_marker, empty_value = path_template[open_pos + 1:close_pos].split("!")
            if value == empty_value:
                repl = ""
            else:
                repl = notempty_marker + value
        path_template = path_template[:open_pos] + repl + path_template[close_pos + 1:]

    # If there are any brackets left, cut the string before them.
    path_template = path_template.split("{")[0]

    return path_template


def _get_tab(path_template, *args):
    def _get_future_template_segments_regex(repl, conj):
        future_template_segments = list(filter(None, re.sub("{([^}]+)![^}]+}", repl, future_template).split("{}")))
        if len(future_template_segments) == 0:
            return None
        else:
            return re.compile("(" + conj.join(map(re.escape, future_template_segments)) + ")")

    # Find the positions of the bracket pair that we tab.
    tab_open_pos = [m.start() for m in re.finditer("{", path_template)][len(args) - args.count(None)]
    tab_close_pos = path_template.find("}", tab_open_pos)

    # Get the segments (template parts separated by brackets) that occur anywhere after the closing bracket
    # and before the next '/'.
    future_template = path_template[tab_close_pos + 1:].split("/")[0]
    # Build two regexes for later:
    # - Matches the presence of any of the required and optional segments.
    any_opt_fut_templ_seg_regex = _get_future_template_segments_regex(r"{}\1{}", "|")
    # - Matches the presence of all required segments in the correct order.
    all_req_fut_templ_seg_regex = _get_future_template_segments_regex("{}", ".+")

    stub_path = _get_path(path_template, *args)

    tab = set()
    for file in glob(stub_path + "*"):
        file = file.replace("\\", "/")  # in case we're on Windows
        file = file[len(stub_path):]  # remove leading stub path

        # If the file name is missing anny of the required future segments, ignore the file.
        if all_req_fut_templ_seg_regex is not None and not all_req_fut_templ_seg_regex.search(file):
            continue

        if tab_close_pos > tab_open_pos + 1:
            notempty_marker, empty_value = path_template[tab_open_pos + 1:tab_close_pos].split("!")
            if file.startswith(notempty_marker):
                # If the current bracket pair is optional, the notempty marker is however present, strip the
                # nonempty marker and continue matching as if the bracket pair wouldn't be optional.
                file = file[len(notempty_marker):]
            else:
                # If the notempty marker is not specified, add the empty value to the tab result.
                tab.add(empty_value)
                continue

        # Find the point until which the current brackets match.
        if any_opt_fut_templ_seg_regex is None:
            tab.add(file)
        else:
            file_tab_end = any_opt_fut_templ_seg_regex.search(file)
            if file_tab_end is None:
                tab.add(file)
            else:
                tab.add(file[:file_tab_end.start()])

    return sorted(tab)


# === Datasets ================================================

_DATASET_PATH_TEMPLATE = DATA_DIR + "/{}/{}"


def dataset_path(archive_name, dataset_name):
    return _get_path(_DATASET_PATH_TEMPLATE,
                     archive_name, dataset_name)


def dataset_tab(archive_name=None):
    return _get_tab(_DATASET_PATH_TEMPLATE,
                    archive_name)


def dataset(archive_name, dataset_name, X_train=True, y_train=True, X_test=True, y_test=True):
    folder = dataset_path(archive_name, dataset_name)

    X_train_data, y_train_data, X_test_data, y_test_data = None, None, None, None

    if archive_name == "ucr":
        if X_train or y_train:
            X_train_data, y_train_data = _read_ucr(os.path.join(folder, dataset_name + "_TRAIN.tsv"))
        if X_test or y_test:
            X_test_data, y_test_data = _read_ucr(os.path.join(folder, dataset_name + "_TEST.tsv"))

    ret = ()
    if X_train:
        ret += (X_train_data,)
    if y_train:
        ret += (y_train_data,)
    if X_test:
        ret += (X_test_data,)
    if y_test:
        ret += (y_test_data,)
    return ret[0] if len(ret) == 1 else ret


def _read_ucr(file_name):
    # Use pandas instead of np.loadtxt() because pandas recognizes that the class label column is an int column.
    data = pd.read_csv(file_name, sep="\t", header=None)
    y = data.pop(0).values
    X = data.values
    return X, y


# === Prefitted classifiers ===================================

_PREFITTED_PATH_RAW_TEMPLATE = "/{}/{}/{}{#!0}."
_PREFITTED_PATH_TEMPLATE = PREFITTED_DIR + _PREFITTED_PATH_RAW_TEMPLATE


def prefitted_path(archive_name, dataset_name, clf_name, clf_rep):
    ext = "hdf5" if clf_name == "resnet" else "joblib"
    return _get_path(_PREFITTED_PATH_TEMPLATE + ext,
                     archive_name, dataset_name, clf_name, clf_rep)


def prefitted_tab(archive_name=None, dataset_name=None, clf_name=None):
    return _get_tab(_PREFITTED_PATH_TEMPLATE,
                    archive_name, dataset_name, clf_name)


def prefitted(archive_name, dataset_name, clf_name, clf_rep):
    path = prefitted_path(archive_name, dataset_name, clf_name, clf_rep)

    if clf_name == "resnet":
        model = keras.models.load_model(path)
        y_train = dataset(archive_name, dataset_name, X_train=False, X_test=False, y_test=False)
        labels = np.unique(y_train)
        return KerasWrapper(model, labels)
    else:
        return joblib.load(path)


# === Test set predictions ====================================

_TEST_PREDS_PATH_TEMPLATE = TEST_PREDS_DIR + _PREFITTED_PATH_RAW_TEMPLATE + "csv"


def test_preds_path(archive_name, dataset_name, clf_name, clf_rep):
    return _get_path(_TEST_PREDS_PATH_TEMPLATE,
                     archive_name, dataset_name, clf_name, clf_rep)


def test_preds_tab(archive_name=None, dataset_name=None, clf_name=None):
    return _get_tab(_TEST_PREDS_PATH_TEMPLATE,
                    archive_name, dataset_name, clf_name)


def test_preds(archive_name, dataset_name, clf_name, clf_rep):
    path = test_preds_path(archive_name, dataset_name, clf_name, clf_rep)
    return np.loadtxt(path, delimiter=",")


# === Test set prediction metrics =============================

def test_pred_metrics_path():
    return os.path.join(RESULTS_DIR, "test_pred_metrics.csv")


def test_pred_metrics():
    path = test_pred_metrics_path()
    return pd.read_csv(path)


# === Impacts =================================================

_IMPACTS_PATH_RAW_TEMPLATE = "/{}/{}/{}{#!0}/{}/specimen_{}{#!0}"
_IMPACTS_PATH_TEMPLATE = IMPACTS_DIR + _IMPACTS_PATH_RAW_TEMPLATE + ".csv"
_IMPACTS_PER_BGCLASS_PATH_TEMPLATE = IMPACTS_DIR + _IMPACTS_PATH_RAW_TEMPLATE + "_per_bgclass.npy"


def impacts_path(archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx, specimen_rep,
                 *, per_bgclass=False):
    return _get_path(_IMPACTS_PER_BGCLASS_PATH_TEMPLATE if per_bgclass else _IMPACTS_PATH_TEMPLATE,
                     archive_name, dataset_name, clf_name, clf_rep,
                     explainer_name, specimen_idx, specimen_rep)


def impacts_tab(archive_name=None, dataset_name=None, clf_name=None, clf_rep=None, explainer_name=None,
                specimen_idx=None,
                *, per_bgclass=False):
    return _get_tab(_IMPACTS_PER_BGCLASS_PATH_TEMPLATE if per_bgclass else _IMPACTS_PATH_TEMPLATE,
                    archive_name, dataset_name, clf_name, clf_rep,
                    explainer_name, specimen_idx)


def impacts(archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx, specimen_rep,
            *, per_bgclass=False):
    path = impacts_path(archive_name, dataset_name, clf_name, clf_rep,
                        explainer_name, specimen_idx, specimen_rep,
                        per_bgclass=per_bgclass)
    return np.load(path) if per_bgclass else np.loadtxt(path, delimiter=",")


# === Impact similarities =====================================

def impact_similarities_path():
    return os.path.join(RESULTS_DIR, "impact_similarities.csv")


def impact_similarities():
    path = impact_similarities_path()
    return pd.read_csv(path)


# === Evaluation data =========================================

_EVALUATION_DATA_TEMPLATE = EVALUATION_DATA_DIR + _IMPACTS_PATH_RAW_TEMPLATE + "_{}.joblib"


def evaluation_data_path(archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx, specimen_rep,
                         evaluator_name):
    return _get_path(_EVALUATION_DATA_TEMPLATE,
                     archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                     specimen_idx, specimen_rep, evaluator_name)


def evaluation_data_tab(archive_name=None, dataset_name=None, clf_name=None, clf_rep=None, explainer_name=None,
                        specimen_idx=None, specimen_rep=None):
    return _get_tab(_EVALUATION_DATA_TEMPLATE,
                    archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx, specimen_rep)


def evaluation_data(archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx, specimen_rep,
                    evaluator_name):
    path = evaluation_data_path(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                                specimen_idx, specimen_rep, evaluator_name)
    return joblib.load(path)


# === Evaluation metrics ======================================

def evaluation_metrics_path():
    return os.path.join(RESULTS_DIR, "evaluation_metrics.csv")


def evaluation_metrics():
    path = evaluation_metrics_path()
    return pd.read_csv(path)
