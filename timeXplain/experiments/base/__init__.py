import os as _os
import sys as _sys
import warnings as _warnings

_warnings.filterwarnings("ignore", category=FutureWarning, module=".*tensorflow.*")

from . import classifiers, parallel, script_utils
from .consts import (ARCHIVE_NAMES,
                     CLASSIFIER_NAMES_TO_LABELS, CLASSIFIER_NAMES,
                     EXPLAINER_NAMES_TO_LABELS, EXPLAINER_NAMES, TIME_EXPLAINER_NAMES,
                     EVALUATOR_NAMES)
from .exfiles import (DATA_DIR, PREFITTED_DIR, RESULTS_DIR, TEST_PREDS_DIR, IMPACTS_DIR, EVALUATION_DATA_DIR,
                      dataset_path, dataset_tab, dataset,
                      prefitted_path, prefitted_tab, prefitted,
                      test_preds_path, test_preds_tab, test_preds,
                      test_pred_metrics_path, test_pred_metrics,
                      impacts_path, impacts_tab, impacts,
                      impact_similarities_path, impact_similarities,
                      evaluation_data_path, evaluation_data_tab, evaluation_data,
                      evaluation_metrics_path, evaluation_metrics)
from .rng import reproducible_rng


def _legacy_module(old_module, new_module):
    _sys.modules[old_module] = _sys.modules[new_module]


# Support for legacy pickles which still expect the old root experiment_base module.
_legacy_module("experiment_base", __name__)

# Support for legacy pickles which still expect old sktime modules.
_legacy_module("sktime.classifiers.elastic_ensemble", "sktime.classifiers.distance_based.elastic_ensemble")
_legacy_module("sktime.classifiers.time_series_neighbors", "sktime.classifiers.distance_based.time_series_neighbors")

# Support for legacy pickles which expect the RotationForest submodule in the root directory.
_legacy_module("rotation_forest", "lib.RotationForest.rotation_forest")

# Disable OMP logging spam on console.
_os.environ["KMP_AFFINITY"] = "noverbose"

# Disable repeated deprecation warnings since sktime causes quite a lot of these.
_warnings.simplefilter("once", category=DeprecationWarning)
