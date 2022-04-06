import numpy as np
import math
import os



class DataLoader():
    def __init__(self, ds_dir, dataset):
        """ Get Explanation weights produced by a model for univariate time series
        """
        self.ds_dir = ds_dir
        self.dataset = dataset
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        org_ds = ['CBF', 'CMJ', 'Coffee', 'ECG200', 'GunPoint']
        s = '' if self.dataset in  org_ds else '.tsv'
        sep =',' if self.dataset in org_ds  else '\t'
        train_file =  os.path.join(self.ds_dir, self.dataset, self.dataset+'_TRAIN'+s)
        test_file  = os.path.join(self.ds_dir, self.dataset, self.dataset+'_TEST'+s)

        train_data = np.genfromtxt(train_file,delimiter=sep)
        self.X_train = np.expand_dims(train_data[:,1:], 1)
        self.y_train = train_data[:,0]

        test_data = np.genfromtxt(test_file, delimiter=sep)
        self.X_test = np.expand_dims(test_data[:,1:], 1)
        self.y_test = test_data[:,0]

    def get_X_y(self,):
        return self.X_train, self.y_train, self.X_test, self.y_test

def get_random_explanation(X_test):
    explanation = np.random.uniform(0,100,size=X_test.shape)
    return explanation


def get_saved_explanation(dataset='CMJ',explanation_method ='MrSEQL-SM'):
    method = explanation_method
    LIME_explanation = ['MrSEQL-LIME', 'Rocket-LIME']
    if method == 'MrSEQL-SM':
        test_weight_file = 'output/explanation_weight/weights_MrSEQL_%s.txt' % ds
    elif method == 'MrSEQL-LIME':    
        test_weight_file = 'output/explanation_weight/weights_LIME_%s.txt' % ds
    elif c == 'ResNetCAM':      
        test_weight_file = 'output/resnet_weights/ResNet_%s_BestModel.hdf5_model_weights.txt' % ds
    else: 
        print('ERROR')
        return

    explanation = np.genfromtxt(test_weight_file, delimiter = ',')

    # Convert from LIME explanation (time-slice level) to general explanation (time-step level)
    if method in LIME_explanation:
        explanation = np.repeat(LIME_explanation, X_test.shape[-1]//10).reshape(X_test.shape[0],-1)
        if explanation.shape[-1] != X_test.shape[-1]: #recalibrate LIME explanation
            last_step_explanation = np.transpose(LIME_explanation)[-1].reshape(-1,1)
            n_pad = X_test.shape[-1] - explanation.shape[-1]
            padded_array = np.repeat(last_step_explanation, n_pad)
            explanation = np.append(explanation, padded_array, axis=-1)

    return explanation
