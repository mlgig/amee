import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from .Noise import Noise
from utils.new.Experiment import Evaluate
import utils.new.visualization as vis



class CompareExplanation():
    def __init__(self,X_train,y_train,X_test,y_test,
        explanation_list, explanation_names, referee_list,
        dataset_name='Coffee', noise_type='zero', include_random_explanation=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.referee_list = referee_list
        self.explanation_names = explanation_names
        self.explanation_list = explanation_list
        if include_random_explanation==True: 
            self.random_explanation = np.squeeze(np.random.uniform(0,100,size=self.X_test.shape))
            self.explanation_list = explanation_list.append(self.random_explanation)
            self.explanation_names = explanation_names.append('random')

        else:
            self.explanation_names = explanation_names
            self.explanation_list = explanation_list
        

        col_names=['dataset','noise_type','XAI_method', 'Referee', 'threshold','metrics: acc']
        self.df = pd.DataFrame(columns=col_names)

        col_names = ['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc']
        auc_df = pd.DataFrame(columns=col_names)
        for ref in self.referee_list:
            for exp,name in zip(self.explanation_list, self.explanation_names):
                evaluate = Evaluate(X_train,y_train,X_test,y_test,exp,referee=ref, noise_type=self.noise_type)
                # print('Explanation AUC for ' + name + ':' + str(evaluate.explanation_auc))
                self.df = evaluate.record_result(dataset_name=self.dataset_name,explanation_name=name, existing_df=True, df=self.df)
                auc_df = auc_df.append({'dataset': self.dataset_name,
                'noise_type':self.noise_type,
                'XAI_method': name,
                'Referee': ref,
                'metrics: explanation_auc': evaluate.explanation_auc}, ignore_index=True)
                

        self.auc_df = auc_df
    def visualize(self,):
        vis.visualize_result(self.df)

    def statistics(self,):
        pass
        


