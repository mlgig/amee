import numpy as np

from experiments.base.exfiles import dataset_tab

ARCHIVE_NAMES = np.array(dataset_tab())

CLASSIFIER_NAMES_TO_LABELS = {
    "rotation_forest": "RotF", "svm_linear": "SVM/Lin", "svm_gak": "SVM/GAK",
    "resnet": "ResNet", "elastic_ensemble": "EE", "proximity_forest": "ProxF",
    "time_series_forest": "TSF", "rise": "RISE", "shapelet_transform": "ST",
    "sax_vsm": "SAX-VSM", "boss": "BOSS", "weasel": "WEASEL"
}

CLASSIFIER_NAMES = list(CLASSIFIER_NAMES_TO_LABELS.keys())

EXPLAINER_NAMES_TO_LABELS = {
    "shap_timeslice_local_mean": "timeXplain Time Slice w/ Local Mean",
    "shap_timeslice_local_mean_bgcs": "timeXplain Time Slice w/ Local Mean + BGCS",
    "shap_timeslice_global_mean": "timeXplain Time Slice w/ Global Mean",
    "shap_timeslice_global_mean_bgcs": "timeXplain Time Slice w/ Global Mean + BGCS",
    "shap_timeslice_local_noise": "timeXplain Time Slice w/ Local Noise",
    "shap_timeslice_local_noise_bgcs": "timeXplain Time Slice w/ Local Noise + BGCS",
    "shap_timeslice_global_noise": "timeXplain Time Slice w/ Global Noise",
    "shap_timeslice_global_noise_bgcs": "timeXplain Time Slice w/ Global Noise + BGCS",
    "shap_timeslice_sample": "timeXplain Time Slice w/ Sample",
    "shap_timeslice_sample_bgcs": "timeXplain Time Slice w/ Sample + BGCS",
    "shap_freqslice_firls": "timeXplain Freq. Slice w/ FIRLS",
    "shap_freqslice_local_mean": "timeXplain Freq. Slice w/ Local Mean Patch",
    "shap_freqslice_local_mean_bgcs": "timeXplain Freq. Slice w/ Local Mean Patch + BGCS",
    "shap_freqslice_sample": "timeXplain Freq. Slice w/ Sample Patch",
    "shap_freqslice_sample_bgcs": "timeXplain Freq. Slice w/ Sample Patch + BGCS",
    "shap_statistic_global": "timeXplain Statistic w/ Global",
    "shap_statistic_global_bgcs": "timeXplain Statistic w/ Global + BGCS",
    "shap_statistic_sample": "timeXplain Statistic w/ Sample",
    "shap_statistic_sample_bgcs": "timeXplain Statistic w/ Sample + BGCS",
    "tree_shap": "Tree SHAP",
    "neural_cam": "NN CAM",
    "shapelet_superpos": "Shapelet Superpos",
    "shapelet_superpos_divide_spread": "Shapelet Superpos + Ds",
    "sax_vsm_word_superpos": "SAX-VSM Word Superpos",
    "sax_vsm_word_superpos_divide_spread": "SAX-VSM Word Superpos + Ds",
    "weasel_word_superpos": "WEASEL Word Superpos",
    "weasel_word_superpos_divide_spread": "WEASEL Word Superpos + Ds"
}

EXPLAINER_NAMES = list(EXPLAINER_NAMES_TO_LABELS.keys())
TIME_EXPLAINER_NAMES = [expl for expl in EXPLAINER_NAMES if "freq" not in expl and "statistic" not in expl]

EVALUATOR_NAMES = ["fidelity", "informativeness"]
