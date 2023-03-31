import os
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import date
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from .Noise import Noise
from .Explanation import load_explanation, create_random_explanation
from .Experiment import Evaluate, train_referee
import utils.visualization as vis
from utils.data import LocalDataLoader



class CompareExplanation():
    def __init__(self, datapath, dataset, explanation_list, explanation_names, 
        referee_list, noise_list, evaluation_on_subset, eval_size):
        self.datapath = datapath
        self.dataset = dataset
        data = LocalDataLoader(datapath,self.dataset)
        self.X_train,self.y_train,self.X_test,self.y_test = data.get_X_y()
        self.referee_list = referee_list
        self.noise_list = noise_list
        self.explanation_names = explanation_names
        self.explanation_list = explanation_list
        self.evaluation_on_subset = evaluation_on_subset
        self.eval_size = eval_size
        

        col_names=['dataset','noise_type','XAI_method', 'Referee', 'threshold','metrics: acc']
        self.acc_df = pd.DataFrame(columns=col_names)

        col_names = ['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc']
        auc_df = pd.DataFrame(columns=col_names)
        for ref in self.referee_list:
            model,transformer = train_referee(self.X_train,self.y_train,ref,self.dataset)
            for noise_type in noise_list:
                for exp,name in zip(self.explanation_list, self.explanation_names):
                    evaluate = Evaluate(datapath=self.datapath,dataset=self.dataset,
                        explanation=exp, referee=ref, noise_type=noise_type,model=model,
                        transformer=transformer, 
                        evaluation_on_subset=self.evaluation_on_subset, eval_size=self.eval_size)
                    # print('Explanation AUC for ' + name + ':' + str(evaluate.explanation_auc))
                    self.acc_df = evaluate.record_result(explanation_name=name, existing_df=True, 
                        df=self.acc_df)
                    auc_df = auc_df.append({'dataset': self.dataset,
                    'noise_type':noise_type,
                    'XAI_method': name,
                    'Referee': ref,
                    'metrics: explanation_auc': evaluate.explanation_auc}, ignore_index=True)
                    

        self.auc_df = auc_df

    def visualize(self,noise_type='global_mean'):
        df1 = self.acc_df.loc[self.acc_df['noise_type'] == noise_type]
        path = './plot/acc_curve_%s_%s' %(self.dataset,noise_type)
        vis.visualize_experiment_result(df1,savefig=True, savepath=path)

    def statistics(self,):
        pass
        


def run_experiment(datapath,dataset,xais,explanation_names,
    lime_xais_list = ['lime','LIME','Lime','ROCKET','ROCKET_SHAP','ROCKET_SHAP_NOSCALER','MRSEQL_SHAP',],
    referee_list = ['rocket','resnet','knn','MrSEQLClassifier','weasel'],
    perturbation_types=['global_mean','local_mean'], 
    include_random_exp=True, 
    evaluation_on_subset=False, 
    eval_size=0.2):
    today = str(date.today()).replace('-', '')

    if not xais:
        xais = []    
    if not explanation_names:
        explanation_names = []
    assert len(xais) == len(explanation_names)
    random_seeds = [2020]
    lime_xais = lime_xais_list
    
    explanation_list = []
    for xai in xais:
        is_reshape=True if xai in lime_xais else False
        weight = load_explanation(datapath=datapath,dataset=dataset,explanation_type=xai, 
            evaluation_on_subset=evaluation_on_subset, reshape_lime=is_reshape,
            eval_size=eval_size)
        explanation_list.append(weight)

    if include_random_exp:
        for seed in random_seeds:
            random_weight, random_weight_name = create_random_explanation(datapath=datapath,
                dataset=dataset,seed=seed)
            explanation_list.append(random_weight)
            explanation_names.append(random_weight_name)

    print('Explanation Shape: ', explanation_list[0].shape)
    print('Total numbers of XAI Methods: ', len(explanation_list))
    print('Names of XAI Methods: ', explanation_names)
    print('Referees: ', referee_list)


    # Compare explanations
    compare = CompareExplanation(datapath, dataset,explanation_list, explanation_names, 
        referee_list, perturbation_types, evaluation_on_subset,eval_size)
    auc_df=compare.auc_df
    acc_df=compare.acc_df
    auc_path = './output/%s_%s.csv' %(dataset,today)
    acc_path = './output/acc_%s_%s.csv' %(dataset,today)
    auc_df.to_csv(auc_path, index=False)
    acc_df.to_csv(acc_path, index=False)
    # compare.visualize(noise_type='local_mean')
    # compare.visualize(noise_type='global_mean')


    return

