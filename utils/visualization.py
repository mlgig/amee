import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import preprocessing
from scipy.interpolate import interp1d

from scripts.Explanation import load_explanation
from .data import LocalDataLoader
from .process_result import get_pos_saliency, index_to_label
from .utils import get_random_color, visualize_class_ts

def visualize_synthetic_data(datapath='data',
    salient_regions=['SmallMiddle','RareTime'],
    processes = ['CAR', 'NARMA','Harmonic', 'PseudoPeriodic', 'GaussianProcess']):
    
    fsize,padsize,legendsize= 25,25,10
    color = ['red','blue']
    nr,nc = len(salient_regions), len(processes)

    fig, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(4*nc,4*nr))
    # axes.tick_params(axis='both', which='major', labelsize=20)
    # axes.tick_params(axis='both', which='minor', labelsize=20)


    for i, region in enumerate(salient_regions):
        for j, process in enumerate(processes):
            dataset = '_'.join((region, process))
            data = LocalDataLoader(datapath,dataset)
            _,_,X,y = data.get_X_y()
            X = np.squeeze(X)
            x_axis = np.arange(0,X.shape[-1],1)
            class_name = list(np.unique(y))
            
            index_dict = defaultdict(list)
            for idx,y0 in enumerate(y):
                index_dict[y0].append(idx)
            
            for class_, c in zip(class_name,color):
                for k in index_dict[class_]:
                    axes[i,j].plot(x_axis,X[k,:], color=c)
            axes[i,j].set_yticklabels(label ,size=fsize)
#       
            if i==0: axes[i,j].set_title(processes[j], fontsize=fsize, pad=padsize) 
            if j==0: 
                axes[i,j].set_ylabel(salient_regions[i], fontsize=fsize, labelpad=padsize)
            if i==nr-1:
                axes[i,j].set_xlabel('Time Step', fontsize=20)
            # axes.xticks(fontsize=25)
    plt.tight_layout()
    plt.savefig('img/synthetic_data.png', dpi=300, )
  
def _visualize_explanation(idx, X_series, explanation,ds, savefig=False):
    """Visualize one time series with explanation by a heatmap
    Args:
        idx: Index of the example to produce heatmap (0-indexed)
        X_series: the X_series that needs to visualize (2d array)
        explanation: coressponding explanation weights for the X_series
        ds: the name of the dataset to explain (for annotation purpose only)

    Return: a plot of heatmap explanation for an example index in a given dataset
    """
    def transform(X):
        ma,mi = np.max(X), np.min(X)
        X = (X - mi)/(ma-mi)
        return X*100
    weight = abs(explanation[idx])
    weight = transform(weight)
    ts = np.squeeze(X_series[idx])
        
    max_length1, max_length2 = len(ts),10000 #
    x1 = np.linspace(0,max_length1,num = max_length1)
    x2 = np.linspace(0,max_length1,num = max_length2)
    y1 = ts
    
    f = interp1d(x1, y1) # interpolate time series
    fcas = interp1d(x1, weight) # interpolate weight color
    weight = fcas(x2) # convert vector of original weight vector to new weight vector

    plt.scatter(x2,f(x2), c = weight, cmap = 'jet', marker='.', s= 1,vmin=0,vmax = 100)
    # plt.xlabel('Explanation for index %d, dataset %s' %(idx, ds))
    cbar = plt.colorbar(orientation = 'vertical')
    
    if savefig:
        plt.savefig('temp.pdf',format='pdf',dpi=300)
    else: plt.show()


def visualize_single_explanation(x, w, savefig=False):
    """Visualize one time series with explanation by a heatmap
    Args:
        idx: Index of the example to produce heatmap (0-indexed)
        X_series: the X_series that needs to visualize (2d array)
        explanation: coressponding explanation weights for the X_series
        ds: the name of the dataset to explain (for annotation purpose only)

    Return: a plot of heatmap explanation for an example index in a given dataset
    """
    def transform(X):
        ma,mi = np.max(X), np.min(X)
        X = (X - mi)/(ma-mi)
        return X*100
    weight = abs(w)
    weight = transform(weight)
    # z = np.histogram(weight)
    # plt.hist(weight, bins = [0,20,40,60,80,100]) 
    # plt.title("histogram") 
    # plt.show()
    ts = np.squeeze(x)
    # print(ts.shape)    
    max_length1, max_length2 = ts.shape[0],10000 #
    x1 = np.linspace(0,max_length1,num = max_length1)
    x2 = np.linspace(0,max_length1,num = max_length2)
    y1 = ts
    
    f = interp1d(x1, y1) # interpolate time series
    fcas = interp1d(x1, weight) # interpolate weight color
    weight = fcas(x2) # convert vector of original weight vector to new weight vector

    plt.scatter(x2,f(x2), c = weight, cmap = 'jet', marker='.', s= 1,vmin=0,vmax = 100)
    # plt.xlabel('Explanation for index %d, dataset %s' %(idx, ds))
    cbar = plt.colorbar(orientation = 'vertical')
    
    if savefig:
        plt.savefig('temp.pdf',format='pdf',dpi=300)
    else: plt.show()


def visualize_experiment_result(df, fsize=15, padsize=15, legendsize=8,savefig=False,savepath='./plot/temp'):
    referees = list(set(df['Referee']))
    xais = list(set(df['XAI_method']))
    datasets = list(set(df['dataset']))
    color = get_random_color(len(xais))
    marker = ['v', 'o', 'd','v','o','d']
    nr,nc = len(datasets),len(referees)
    x = np.arange(0,101,10)

    if nr==1 and nc==1: # one XAI, one dataset --> single figure
        fig = plt.figure(figsize=(6,4))
        ref= referees[0]
        dataset=datasets[0]
        print(ref)
        for xai,c,m in zip(xais,color,marker):
            y = df[(df['Referee'] == ref) & 
                                  (df['XAI_method'] == xai) & 
                                  (df['dataset'] == dataset)]['metrics: acc']
            plt.plot(x,y, color=c, marker=m)
            plt.title('Referee: %s' %ref.upper(), fontsize=fsize)
            plt.xlabel('Noise Level in Percentage')
            plt.ylabel('Dataset: %s' %dataset, fontsize=fsize, labelpad=padsize)
            plt.legend(xais, loc='upper right', fontsize=legendsize)
        plt.show()
    
    else:
        fig, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(4*nc,4*nr))
        for i, dataset in enumerate(datasets):
            for j, ref in enumerate(referees):
                for xai,c,m in zip(xais,color,marker):
                    y = df[(df['Referee'] == ref) & 
                                      (df['XAI_method'] == xai) & 
                                      (df['dataset'] == dataset)]['metrics: acc']
                    if   nr==1: # ONE dataset only --> one row of XAIs
                        axes[j].plot(x,y, color=c, marker = m)
                        axes[j].set_title('Referee: %s' %ref.upper(), fontsize=fsize, pad=padsize)
                        axes[j].set_xlabel('Noise Level in Percentage')
                        if j==0: 
                            axes[j].set_ylabel('Dataset: %s' %dataset, fontsize=fsize, labelpad=padsize)
                            axes[j].legend(xais, loc='upper right', fontsize=legendsize)

                    elif nc==1: # ONE referee only --> one column of datasets
                        axes[i].plot(x,y, color=c, marker = m)
                        axes[i].set_ylabel('Dataset: %s' %dataset, fontsize=fsize, labelpad=padsize)
                        if i==0: 
                            axes[i].set_title('Referee: %s' %ref.upper(), fontsize=fsize, pad=padsize)
                            axes[i].legend(xais, loc='upper right', fontsize=legendsize)
                        if i == len(datasets)-1:
                            axes[i].set_xlabel('Noise Level in Percentage')

                    else:
                        axes[i,j].plot(x,y, color=c, marker = m)
                        if i==0: axes[i,j].set_title('Referee: %s' %ref.upper(), fontsize=fsize, pad=padsize) 
                        if j==0: 
                            axes[i,j].set_ylabel('Dataset: %s' %dataset, fontsize=fsize, labelpad=padsize)
                            axes[i,j].legend(xais, loc='upper right', fontsize=legendsize)
                        if i == len(datasets)-1:
                            axes[i,j].set_xlabel('Noise Level in Percentage')
    
        plt.tight_layout()
  
        if savefig:
            plt.savefig(savepath+'.png',bbox_inches='tight', pad_inches=0)


def visualize_groundtruth(datapath,dataset,index):
    data = LocalDataLoader(datapath,dataset)
    X_train,y_train,X_test,y_test = data.get_X_y()
    
    data = np.load('data/synth/'+dataset+'_TEST_meta.npy')
    saliency_index = data[:,1:3]
    saliency = index_to_label(saliency_index)
    visualize_single_explanation(X_test[index],saliency[index])


def visualize_explainer(datapath,dataset,index, xai = 'GradientShap', 
                        lime_xais_list = ['lime','LIME','Lime','ROCKET','ROCKET_SHAP']):
    data = LocalDataLoader(datapath,dataset)
    X_train,y_train,X_test,y_test = data.get_X_y()
    
    is_reshape=True if xai in lime_xais_list else False
    saliency = load_explanation(datapath,dataset,explanation_type=xai,
                            reshape_lime=is_reshape,
                           evaluation_on_subset=False,
                           eval_size=None)
    visualize_single_explanation(X_test[index],saliency[index])

def visualize_masking_type(X, explanation,threshold=30,plot_index=70):
    dim1, dim2 = ['local', 'global'], ['mean', 'gaussian']
    nr,nc = len(dim1), len(dim2)
    X = np.squeeze(X)
    fsize,padsize,legendsize,linewidth= 15,10,8,3
    
    fig, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(6*nc,3*nr))
    for i, x in enumerate(dim1):
        for j, y in enumerate(dim2):
            noise_type = '_'.join((x, y))
            X = np.squeeze(X)
            noise = Noise(X=X,explanation=explanation, noise_type=noise_type)
            noise.add_noise(threshold=threshold)
            axes[i,j].plot(noise.X_perturbed_2d[plot_index], linewidth=linewidth)
            axes[i,j].plot(noise.X_test[plot_index], linewidth=linewidth)
            if i==0: axes[i,j].set_title(dim2[j].capitalize(), fontsize=fsize, pad=padsize) 
            if j==0: 
                axes[i,j].set_ylabel(dim1[i].capitalize(), fontsize=fsize, labelpad=padsize)
            if i==0 and j==0:
                axes[i,j].legend(['Noisy','Normal'], loc='upper left', fontsize=legendsize)
            axes[i,j].set_xlabel('Time Step')
            

            
            