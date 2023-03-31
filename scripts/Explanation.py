import numpy as np
import os
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from lime import explanation
from lime import lime_base
from utils.lime_timeseries import LimeTimeSeriesExplanation
import utils.data as utils_data
from utils.data import LocalDataLoader


_LIME_NUM_SAMPLE = 100
_LIME_num_features = 10
_LIME_num_slices = 10


class GetExplanation():
    def __init__(self, datapath="./data", dataset='CMJ', explanation_type='mrseql'):
        """ Get Explanation weights produced by a model for univariate time series
        """
        self.dataset = dataset
        self.datapath = datapath
        data = LocalDataLoader(self.datapath,self.dataset)
        self.X_train,self.y_train,self.X_test,self.y_test = data.get_X_y()
        self.explanation_type = explanation_type
        self.X_test_2d = np.squeeze(self.X_test)
    
        if   self.explanation_type == 'mrseql': self.explanation_weight = self.get_weight_mrseql()
        elif self.explanation_type == 'rocket': self.explanation_weight = self.get_weight_rocket()
        elif self.explanation_type == 'lime':   self.explanation_weight = self.get_weight_lime_mrseql()
        elif self.explanation_type == 'ridgecv': self.explanation_weight = self.get_weight_ridgecv()
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

        weights = features
        # weights = reshape_lime_explanation(self.datapath,self.dataset,features)
        return weights
    def get_weight_rocket(self,):
        rocket = Rocket()  
        rocket.fit(self.X_train)
        X_train_transform = rocket.transform(self.X_train)
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        model.fit(X_train_transform, self.y_train)
        num_sample = len(self.y_test)
        num_classes = np.unique(self.y_test).shape[0]
        class_names = [str(int(x)) for x in np.unique(self.y_test)]
        explainer = LimeTimeSeriesExplanation(class_names=class_names,feature_selection='auto')
        features = np.empty(dtype=float, shape=(0,_LIME_num_features))

        def distance_fn_rocket(X):
            X_transform = rocket.transform(X)
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

        weights = features
        # weights = reshape_lime_explanation(self.datapath,self.dataset,features)
        return weights
    def get_weight_ridgecv(self,):
        X_train, X_test = np.squeeze(self.X_train), np.squeeze(self.X_test)
        model = RidgeClassifierCV(alphas=np.logspace(-2, 3, 10), normalize=True)
        model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)
        coefs = model.coef_
        x = model.decision_function(X_test)
        le = preprocessing.LabelEncoder()
        le.fit(y_pred)
        y_pred_transform = le.transform(y_pred)
        weights = np.empty(dtype=float, shape=(0,X_test.shape[-1]))
        
        for i, ts in enumerate(X_test):
            pred_class = int(y_pred_transform[i]) if model.classes_[-1] > 1 else 0 # this variable just means the relevant index to get the weight
            
            if model.classes_[-1] == 1 and y_pred[i] == 0: # binary classification + class 0 predicted
                sign_ = -1
            else:
                sign_ = 1
            w = X_test[i] * coefs * sign_ 
            w = [w[pred_class]]
            weights = np.append(weights, w, axis = 0)
    

        return weights
    def save_explanation_weight(self,ds='CMJ'):
        fileName = 'output/explanation_weight/{}/{}.txt'.format(ds,explanation_type)
        np.savetxt(fileName, self.explanation_weight, delimiter=",")



########################################################
def reshape_lime_explanation(datapath,dataset, LIME_explanation, evaluation_on_subset, eval_size):
    """ Reshape a patch-like explanation to the same dimension as original time series
    """
    data = LocalDataLoader(datapath,dataset)
    _,_,X_test,y_test = data.get_X_y()
    if evaluation_on_subset:
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=eval_size, random_state=9000)
    # print(LIME_explanation.shape)
    # print(X_test.shape)
    explanation = np.repeat(LIME_explanation, X_test.shape[-1]//10).reshape(X_test.shape[0],-1)
    if explanation.shape[-1] != X_test.shape[-1]: #recalibrate LIME explanation
        last_step_explanation = np.transpose(LIME_explanation)[-1].reshape(-1,1)
        n_pad = X_test.shape[-1] - explanation.shape[-1]
        padded_array = np.repeat(last_step_explanation, n_pad, axis=-1)
        explanation = np.append(explanation, padded_array, axis=-1)
    return explanation

def load_explanation(datapath,dataset,explanation_type, evaluation_on_subset,reshape_lime,eval_size):
    xai = explanation_type
    # path = 'exp_weights/%s/%s.csv' % (dataset,xai)
    path = 'exp_weights/weights_%s_%s.txt' % (xai,dataset)
    isExist = os.path.exists(path)
    if not isExist:
        print('File path not exists')
    weight = np.genfromtxt(path, delimiter = ',')
    if reshape_lime:
        weight=reshape_lime_explanation(datapath,dataset,weight, evaluation_on_subset,eval_size)
    return weight


def create_random_explanation(datapath,dataset,seed=2020):
    np.random.seed(seed)
    data = LocalDataLoader(datapath,dataset)
    _,_,X_test,_ = data.get_X_y()
    random_weight = np.squeeze(np.random.uniform(0,100,size=X_test.shape))
    # print(random_weight)
    name = 'random'+str(seed)
    return random_weight, name