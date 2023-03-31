# timeXplain

This is the soon to be published Python 3.7 reference implementation of the timeXplain framework.
At the moment, there is no comprehensive documentation available.
To get started with timeXplain, take a look at the notebooks in `experiments/notebooks/`.
The notebook `case_study.ipynb` reproduces the small case study from the paper and thereby introduces how to use the package.
The notebook `explainers.ipynb` demonstrates all explanation methods offered by this package, both model-agnostic and model-specific ones.

## Setup

For the library itself, install the dependencies listed in `setup.py`, and optionally the dependencies listed in `requirements-optional.txt`. Installing numba is highly recommended as it may significantly improve performance.

Running the demo notebooks requires installing the dependencies in `requirements-experiments.txt`. They have been tested with the library versions listed in that file, but newer versions should also be fine.

## Model-specific explainer compatibility

The model-specific explainers (which are not to be confused with the model-agnostic explainers) were implemented to be compatible with the following libraries:

* SaxVsmWordSuperposExplainer & WeaselExplainer: pyts
* NeuralCamExplainer: keras
* ShapeletTransformExplainer: sktime
* LinearShapExplainer & TreeShapExplainer: refer to linar shap and tree shap for compatibility.

Adaption to other libraries should not be too difficult.
