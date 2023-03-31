import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)



def get_random_color(n):
    ans = []
    random.seed(1)
    for j in range(n):
        rand_color = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        ans.append(rand_color)
    return ans


def create_random_explanation(datapath,dataset,seed=2020):
    data = LocalDataLoader(ds_dir,ds)
    _,_,X_test,_ = data.get_X_y()
    random_weight = np.squeeze(np.random.uniform(0,100,size=X_test.shape))
    name = 'random'+str(seed)
    return random_weight, name


def visualize_class_ts(X,y):
    X = np.squeeze(X)
    class_name = list(np.unique(y))
    num_class = len(class_name)
    color_list = get_random_color(num_class)
    index_dict = defaultdict(list)
    for i,y0 in enumerate(y):
        index_dict[y0].append(i)
    
    nr,nc = 1,len(class_name)

    fig, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(4*nc,4*nr))
        
    for class_,color in zip(class_name,color_list):
        for j in range(X.shape[0]):
            if j in index_dict[class_]:
                axes[int(class_)].plot(X[j,:], color=color)
                axes[int(class_)].set_title('Class %d sample' %class_, fontsize=15, pad=8)
                        
    plt.tight_layout()
  

def load_xais_comparison_output(ds):
    path = glob.glob('./output/%s_*.csv'%(ds))
    assert path, 'No comparison files found.'
    data = [pd.read_csv(p) for p in path]
    auc_df=pd.concat(data, ignore_index=True, axis=0)
    auc_df = auc_df[auc_df['noise_type'] !='original_gaussian']
    return auc_df

# def get_best_method()

def xai_average_ranking(ds,display_detail=False):
    colnames = ['dataset','best','worst']
    df = pd.DataFrame(columns=colnames)

    auc_df = load_xais_comparison_output(ds)
    x = get_final_ranking(auc_df,referees =None ,ranking_by_perturbation_method=False, 
                              beautify_display=False)
    x['ranking'] = x.groupby('dataset')['scaled_ranking'].rank(ascending=True)
    curr_best = x['XAI_method'][x['scaled_ranking'].idxmin()] 
    curr_worst = x['XAI_method'][x['scaled_ranking'].idxmax()]
    df = df.append({'dataset': ds,
                   'best':curr_best,
                    'worst':curr_worst,
                   }, ignore_index=True)
    print(df)
    if display_detail:
        print(x)
    return curr_best, curr_worst

def xai_ranking_by_referee(ds, referee_name):
    auc_df = load_xais_comparison_output(ds)
    x = get_final_ranking(auc_df,referees =[referee_name] ,ranking_by_perturbation_method=False,
                             beautify_display=False)
    x['ranking'] = x.groupby('dataset')['scaled_ranking'].rank(ascending=True)
    print(x)
#   print(x.shape[0])


def save_data_std(X_train,y_train,X_test,y_test,ds='BeepTest2',datapath='data'):
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape, y_test.shape)
    train = np.column_stack((y_train,X_train))
    test = np.column_stack((y_test,X_test)) 
    print(train.shape)
    print(test.shape)

    train_file,test_file = '%s_TRAIN.txt'%ds,'%s_test.txt'%ds
    np.savetxt(train_file,train)
    np.savetxt(test_file,test)