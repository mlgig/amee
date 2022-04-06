import numpy as np
import os
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from lime import explanation
from lime import lime_base
from utils.new.lime_timeseries import LimeTimeSeriesExplanation


_LIME_NUM_SAMPLE = 100
_LIME_num_features = 10
_LIME_num_slices = 10

class GetExplanation():
    def __init__(self, X_train,y_train,X_test,y_test, explanation_type='mrseql'):
        """ Get Explanation weights produced by a model for univariate time series
        """
        self.explanation_type = explanation_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_2d = np.squeeze(self.X_test)
        if   self.explanation_type == 'mrseql': self.explanation_weight = self.get_weight_mrseql()
        elif self.explanation_type == 'minirocket': self.explanation_weight = self.get_weight_minirocket()
        elif self.explanation_type == 'lime':   self.explanation_weight = self.get_weight_lime_mrseql()
    def train_model_mrseql(self,):
        model = MrSEQLClassifier(seql_mode="fs")
        model.fit(self.X_train,self.y_train)
        return model

    def get_weight_mrseql(self,):
        self.model = self.train_model_mrseql()
        
        y_pred = self.model.predict(self.X_test)
        le = preprocessing.LabelEncoder()
        le.fit(y_pred)
        y_pred_transform = le.transform(y_pred)
        weights = np.empty(dtype=float, shape=(0,self.X_test.shape[-1]))
        
        for i, ts in enumerate(self.X_test_2d):
            w = self.model.map_sax_model(ts)
            pred_class = int(y_pred_transform[i])
            w = [w[pred_class]]
            weights = np.append(weights, w, axis = 0)
        return weights

    def get_weight_lime_mrseql(self,):
        self.model = self.train_model_mrseql()

        num_sample = self.y_test.shape[0]
        class_names = [str(int(x)) for x in np.unique(self.y_test)]
        explainer = LimeTimeSeriesExplanation(class_names=class_names, feature_selection='auto')
        features = np.empty(dtype=float, shape=(0,_LIME_num_features))
        # print(self.model.predict_proba(self.X_train).shape)

        for idx in range(num_sample):
            series = self.X_test_2d[idx, :]
            exp = explainer.explain_instance(series, self.model.predict_proba, 
                num_features=_LIME_num_features, num_samples=_LIME_NUM_SAMPLE, 
                num_slices=_LIME_num_slices, 
                replacement_method='total_mean', training_set=self.X_train)
            temp, ans = [], []
            for i in range(_LIME_num_features):
                feature,weight = exp.as_list()[i]
                temp.append((feature,weight))
            temp.sort()
            for _, val in temp: ans.append(val)
            features = np.append(features, np.array([ans]), axis=0)       

        weights = reshape_lime_explanation(self.X_test,self.y_test,features)
        return weights


    def get_weight_minirocket(self,):
        minirocket = MiniRocket()  # by default, MiniRocket uses ~10,000 kernels
        minirocket.fit(self.X_train)
        X_train_transform = minirocket.transform(self.X_train)
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        model.fit(X_train_transform, self.y_train)
        num_sample = len(self.y_test)
        num_classes = np.unique(self.y_test).shape[0]
        class_names = [str(int(x)) for x in np.unique(self.y_test)]
        explainer = LimeTimeSeriesExplanation(class_names=class_names,feature_selection='auto')
        features = np.empty(dtype=float, shape=(0,_LIME_num_features))

        def distance_fn_rocket(X):
            X_transform = minirocket.transform(X)
            ans = model.decision_function(X_transform)
            if num_classes>2: return ans
            else:
                ans = np.expand_dims(ans,1)              
                z1 = np.array(ans<=0)
                z2 = np.array(ans>0)
                z = np.hstack((z1,z2))
                ans = ans*z
                return ans
        for idx in range(num_sample):
            series = self.X_test_2d[idx, :]
            exp = explainer.explain_instance(series, distance_fn_rocket, 
                num_features=_LIME_num_features, num_samples=_LIME_NUM_SAMPLE, num_slices=_LIME_num_slices, 
                replacement_method='total_mean', training_set=self.X_train)
            temp, ans = [], []
            for i in range(_LIME_num_features):
                feature, weight = exp.as_list()[i]
                temp.append((feature,weight))
            temp.sort()
            for _, val in temp: ans.append(val)
            features = np.append(features, np.array([ans]), axis = 0)    

        weights = reshape_lime_explanation(self.X_test,self.y_test,features)
        return weights


    def save_explanation_weight(self,ds='CMJ'):
        fileName = 'output/explanation_weight/{}/{}.txt'.format(ds,explanation_type)
        np.savetxt(fileName, self.explanation_weight, delimiter=",")






########################################################
def reshape_lime_explanation(X_test, y_test, LIME_explanation):
    explanation = np.repeat(LIME_explanation, X_test.shape[-1]//10).reshape(X_test.shape[0],-1)
    if explanation.shape[-1] != X_test.shape[-1]: #recalibrate LIME explanation
        last_step_explanation = np.transpose(LIME_explanation)[-1].reshape(-1,1)
        n_pad = X_test.shape[-1] - explanation.shape[-1]
        padded_array = np.repeat(last_step_explanation, n_pad, axis=-1)
        explanation = np.append(explanation, padded_array, axis=-1)
    return explanation