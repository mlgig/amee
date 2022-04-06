import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.interpolate import interp1d

def visualize_explanation(idx, X_series, explanation,ds):
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
    plt.xlabel('Explanation for index %d, dataset %s' %(idx, ds))
    cbar = plt.colorbar(orientation = 'vertical')
    plt.show()




def visualize_result(df):
    referees = list(set(df['Referee']))
    xais = list(set(df['XAI_method']))
    datasets = list(set(df['dataset']))
    color = ['red', 'blue', 'green','orange']
    marker = ['v', 'o', 'd','v']
    nr,nc =len(datasets), len(referees)
    x = np.arange(0,101,10)

    if nr==1 and nc==1:
        fig = plt.figure()
        ref= referees[0]
        dataset=datasets[0]
        print(ref)
        for xai,c,m in zip(xais,color,marker):
            y = df[(df['Referee'] == ref) & 
                                  (df['XAI_method'] == xai) & 
                                  (df['dataset'] == dataset)]['metrics: acc']
            plt.plot(x,y, color=c, marker=m)
            plt.xlabel('Referee: %s' %ref.upper(), fontsize=13)
            plt.ylabel('Dataset: %s' %dataset, fontsize=13, labelpad=15)
            plt.legend(xais, loc='upper right', fontsize=8)
        plt.show()
    
    else:
        fig, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(8,5))
        for i, dataset in enumerate(datasets):
            for j, ref in enumerate(referees):
                for xai,c,m in zip(xais,color,marker):
                    y = df[(df['Referee'] == ref) & 
                                      (df['XAI_method'] == xai) & 
                                      (df['dataset'] == dataset)]['metrics: acc']
                    if   nr==1: # ONE dataset only 
                        axes[j].plot(x,y, color=c, marker = m)
                        axes[j].set_title('Referee: %s' %ref.upper(), fontsize=13, pad = 15)
                        if j==0: 
                            plt.ylabel('Dataset: %s' %dataset, fontsize=13, labelpad=15)
                            axes[j].legend(xais, loc='upper right', fontsize=8)


                    elif nc==1: # ONE referee only
                        axes[i].plot(x,y, color=c, marker = m)
                        # axes[i].set_ylabel('Dataset: %s' %dataset, fontsize=13, labelpad=15)
                        if i==0: 
                            axes[i].set_ylabel('Dataset: %s' %dataset, fontsize=13, labelpad=15)
                            axes[i].set_title('Referee: %s' %ref.upper(), fontsize=13, pad = 15)
                            axes[i].legend(xais, loc='upper right', fontsize=8)

                    else:
                        axes[i,j].plot(x,y, color=c, marker = m)
                        if i==0: axes[i,j].set_title('Referee: %s' %ref.upper(), fontsize=13, pad = 15) 
                        if j==0: 
                            axes[i,j].set_title('Referee: %s' %ref.upper(), fontsize=13, pad = 15)
                            axes[i,j].legend(xais, loc='upper right', fontsize=8)
                                
    
        plt.tight_layout()
        plt.show()
