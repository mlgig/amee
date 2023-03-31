import numpy as np
import math
import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import get_random_color, visualize_class_ts

_NUM_FEATURES = 50


class LocalDataLoader():
    def __init__(self, datapath="./data", dataset='CMJ',num_features=_NUM_FEATURES):
        """ Load dataset from a local folder
        """
        self.ds_dir = datapath
        self.dataset = dataset
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.num_features = num_features        
    def get_X_y(self,onehot_label=False,synth=False):
        if synth == False:
            s, sep = '.txt', None
            train_file =  './%s/%s/%s_TRAIN%s' %(self.ds_dir,self.dataset,self.dataset,s) 
            test_file = './%s/%s/%s_TEST%s' %(self.ds_dir,self.dataset,self.dataset,s)
            train_data = np.genfromtxt(train_file,delimiter=sep)
            test_data = np.genfromtxt(test_file, delimiter=sep)

            self.X_train = np.expand_dims(train_data[:,1:], 1)
            self.y_train = train_data[:,0]


            self.X_test = np.expand_dims(test_data[:,1:], 1)
            self.y_test = test_data[:,0]

        else:
            self.ds_dir=self.ds_dir+'/synth/'
            train_file = self.ds_dir+self.dataset+'_TRAIN.npy'
            test_file  = self.ds_dir+self.dataset+'_TEST.npy'
            train_label = self.ds_dir+self.dataset+'_TRAIN_meta.npy'
            test_label = self.ds_dir+self.dataset+'_TEST_meta.npy'
            
            train_data = np.load(train_file)
            test_data = np.load(test_file)
            train_label = np.load(train_label)
            test_label = np.load(test_label)
            
            selected_feature = self.num_features//2
            self.X_train = np.expand_dims(train_data[:,:,selected_feature],1)
            self.X_test = np.expand_dims(test_data[:,:,selected_feature],1)
            
            self.y_train = train_label[:,0]
            self.y_test = test_label[:,0]
            


        # Standardize labels 
        encoder = OneHotEncoder(categories='auto', sparse=False)
        self.y_train = encoder.fit_transform(np.expand_dims(self.y_train, axis=-1))
        self.y_test = encoder.transform(np.expand_dims(self.y_test, axis=-1))
        
        if onehot_label == False:
            self.y_train = np.argmax(self.y_train,axis=1)
            self.y_test = np.argmax(self.y_test, axis=1)


        return self.X_train, self.y_train, self.X_test, self.y_test

    
        





    def createTensorDataset(self,batch_size=64):
        self.X_train,self.y_train,self.X_test,self.y_test=self.get_X_y(onehot_label=True)

        train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(),torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test).float(),torch.from_numpy(self.y_test))
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
        return train_loader,test_loader

    def get_loaders(self,mode='train',batch_size=64,val_size=0.2):
        self.X_train,self.y_train,self.X_test,self.y_test=self.get_X_y(onehot_label=True)
        
        self.X_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train)
        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test)

        if mode == 'train':
            assert val_size is not None
            X_train,y_train,X_val,y_val = train_test_split_tensor(
                self.X_train,self.y_train,
                split_size=val_size,
                )

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                )
            val_loader = DataLoader(
                TensorDataset(X_val,y_val),
                batch_size=batch_size,
                shuffle=False,
                )
            return train_loader,val_loader
        
        else:
            test_loader = DataLoader(
                TensorDataset(self.X_test,self.y_test),
                batch_size=batch_size,
                shuffle=False
                )
            return test_loader,None

def train_test_split_tensor(X_tensor,y_tensor,split_size):
    X_train,X_test,y_train,y_test=train_test_split(
        X_tensor.numpy(),y_tensor.numpy(),
        test_size=split_size,
        )
    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)
    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor
  
def data_summary(datapath,dataset):
    print('Dataset: %s' %(dataset))
    data = LocalDataLoader(datapath,dataset)
    X_train,y_train,X_test,y_test = data.get_X_y()

    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ', X_test.shape)
    names = [str(int(x)) for x in np.unique(y_test)]
    print('Unique target class: ', names)

    print('Dataset: %s, Training Data-Global mean value: %2.5f' % (dataset, np.mean(X_test)))
    
    visualize_class_ts(X_train,y_train)

'''
def merge_resplit_data(dataset, datapath="./data", test_size=0.2, random_seed=2023):
    org_ds = ['CBF', 'CMJ', 'Coffee', 'ECG200', 'GunPoint']     
    s = '' if dataset in  org_ds else '.txt'
    sep =',' if dataset in org_ds  else None
    train_file =  './%s/%s/%s_TRAIN%s' %(datapath,dataset,dataset,s) 
    test_file = './%s/%s/%s_TEST%s' %(datapath,dataset,dataset,s)

    train_data = np.genfromtxt(train_file,delimiter=sep)
    test_data = np.genfromtxt(test_file, delimiter=sep)
    merged_data = np.vstack((train_data,test_data))
    train,test = train_test_split(merged_data, test_size=0.2, random_state=2023)

    path = 'newdata/%s' %dataset
    if not os.path.exists(path): os.makedirs(path)
    train_file =  "newdata/%s/%s_TRAIN.txt" %(dataset,dataset)
    test_file = "newdata/%s/%s_TEST.txt" %(dataset,dataset)
    

    np.savetxt(train_file,train)
    np.savetxt(test_file,test)
'''
# Source: https://github.com/mlgig/mrsqm/blob/main/example/util.py
def load_from_arff_to_dataframe(

    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.
    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    has_class_labels: bool
        true then line contains separated strings and class value contains
        list of separated strings, check for 'return_separate_X_and_y'
        false otherwise.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.
    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    
    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, "r", encoding="utf-8") as f:
        for line in f:

            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data["class_vals"] = pd.Series(class_val_list)

    return x_data

