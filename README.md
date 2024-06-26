# AMEE: Robust explainer recommendation for time series classification

Paper access: [10.1007/s10618-024-01045-8](https://link.springer.com/article/10.1007/s10618-024-01045-8)

## Abstract
> Time series classification is a task which deals with temporal sequences, a prevalent data type common in domains such as human activity recognition, sports analytics and general sensing. In this area, interest in explanability has been growing as explanation is key to understand the data and the model better. Recently, a great variety of techniques (e.g., LIME, SHAP, CAM) have been proposed and adapted for time series to provide explanation in the form of saliency maps, where the importance of each data point in the time series is quantified with a numerical value. However, the saliency maps can and often disagree, so it is unclear which one to use. This paper provides a novel framework to quantitatively evaluate and rank explanation methods for time series classification. We show how to robustly evaluate the informativeness of a given explanation method (i.e., relevance for the classification task), and how to compare explanations side-by-side. The goal is to recommend the best explainer for a given time series classification dataset. We propose AMEE, a Model-Agnostic Explanation Evaluation framework, for recommending saliency-based explanations for time series classification. In this approach, data perturbation is added to the input time series guided by each explanation. Our results show that perturbing discriminative parts of the time series leads to significant changes in classification accuracy, which can be used to evaluate each explanation. To be robust to different types of perturbations and different types of classifiers, we aggregate the accuracy loss across perturbations and classifiers. This novel approach allows us to recommend the best explainer among a set of different explainers, including random and oracle explainers. We provide a quantitative and qualitative analysis for synthetic datasets, a variety of time-series datasets, as well as a real-world case study with known expert ground truth.

Please cite as:

```
@article{nguyen2024robustexp,
  title={Robust Explainer Recommendation for Time Series Classification}, 
      author={Thu Trang Nguyen and Thach Le Nguyen and Georgiana Ifrim},
      journal={Data Mining and Knowledge Discovery},
  year={2024},
  publisher={Springer},
  doi = {10.1007/s10618-024-01045-8},
arxivId = {2306.05501}}
```
## Reproducibility
Experiments can be reproduced using the notebook "Compare Explanations.ipynb".

## Requirements

AMEE requires the following to run:

  * [sktime](https://github.com/sktime/sktime) 0.9.0

## Acknowledgements
We would like to thank the anonymous reviewers for their detailed and constructive feedback. We would also like to gratefully acknowledge the work by researchers at University of California Riverside, USA (especially Eamonn Keogh and his team) for their effort in collecting, updating and making available the UCR time series classification benchmarks. We want to thank all researchers in time series classification and explainable AI who have made their data, code and results open source and have helped the reproducibility of research methods in this area. This work was funded by Science Foundation Ireland through the SFI Centre for Research Training in Machine Learning (18/CRT/6183), the Insight Centre for Data Analytics (12/RC/2289_P2) and the VistaMilk SFI Research Centre (SFI/16/RC/3835).
