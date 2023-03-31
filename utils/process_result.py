import numpy as np
import pandas as pd
import seaborn as sns
import glob
from os import listdir
from os.path import isfile, join
from sklearn import metrics
from sklearn import preprocessing
from utils.data import LocalDataLoader
from scripts.Explanation import load_explanation, create_random_explanation



########################################################################## 
#                          SYNTHETIC DATASET                             #
##########################################################################

def get_pos_saliency(X, threshold=0.5):
    """Converts a probability array to prediction array of 0,1 with specified threshold
    """
    X_scaled = (X - X.min(axis=1,keepdims=True)) / (X.max(axis=1,keepdims=True) - X.min(axis=1,keepdims=True))
    return X_scaled>=threshold

def index_to_label(start_end_array, n_steps=50):
    """Converts from true saliency with start and end values to a 0,1 numpy array
    """
    ans = list(np.zeros(shape=(start_end_array.shape[0],n_steps)))
    start_end_array = list(start_end_array)
    for i, val in enumerate(start_end_array):
        start, end = int(val[0]), int(val[1])
        for j in range(start,end):
            ans[i][j] = 1
        
    return np.array(ans)



def precision_recall_synthetic(datapath='data',metric='precision', threshold=0.5):
    ds0 = SYNTHETIC_DS[0]
    df = _get_metric(datapath,ds0,threshold=threshold)


    for dataset in SYNTHETIC_DS[1:]:
        df1 = _get_metric(datapath,dataset,threshold=threshold)
        # print(df1)
        df = df.append(df1)
    # print(df)

    table = pd.pivot_table(df, values=metric, 
        index=['dataset'],
        columns=['XAI_method'],
         aggfunc=np.sum)

    return table

        


def _get_metric(datapath,dataset,threshold, lime_xais_list=['lime','LIME','Lime','ROCKET','ROCKET_SHAP','MRSEQL_SHAP']):
    
    data = LocalDataLoader(datapath,dataset)
    X_train,y_train,X_test,y_test = data.get_X_y()

    data = np.load('data/synth/'+dataset+'_TEST_meta.npy')
    true_saliency_index = data[:,1:3]
    true_saliency = index_to_label(true_saliency_index)
    
    true_saliency = (np.rint(true_saliency)).astype(int)
    # print(np.unique(true_saliency))
    colnames = ['dataset','XAI_method','precision','recall', 'f1_score']
    df = pd.DataFrame(columns=colnames)

    for xai, name in zip(SYNTHETIC_XAIS_LIST, SYNTHETIC_XAIS_NAMES):
        # print(xai)
        if xai == 'random':
            explainer_saliency,_ = create_random_explanation(datapath=datapath,dataset=dataset)
            
        else:
            is_reshape=True if xai in lime_xais_list else False
            explainer_saliency = load_explanation(datapath,dataset,explanation_type=xai,
                            reshape_lime=is_reshape,
                           evaluation_on_subset=False,
                           eval_size=None)

        explainer_saliency = abs(explainer_saliency)
        explainer_saliency =  explainer_saliency / explainer_saliency.max(axis=1).reshape(-1,1)
    
        explainer_saliency = (explainer_saliency>threshold)*1
        # print("explainer saliency",np.unique(explainer_saliency))
        
        prec = metrics.precision_score(true_saliency ,explainer_saliency, average='micro')
        rec = metrics.recall_score(true_saliency ,explainer_saliency, average='micro')
        f1_score = metrics.f1_score(true_saliency ,explainer_saliency, average='micro')

        df = df.append({'dataset': dataset,
                        'XAI_method': name,
                       'precision':prec,
                        'recall':rec,
                        'f1_score':f1_score,
                       }, ignore_index=True)

    return df

########################################################################## 
#                             UCR DATASET                                #
##########################################################################



def load_xais_comparison_output(ds):
    path = glob.glob('./output/%s_*.csv'%(ds))
    assert path, 'No comparison files found.'
    data = [pd.read_csv(p) for p in path]
    auc_df=pd.concat(data, ignore_index=True, axis=0)
    auc_df = auc_df[auc_df['noise_type'] !='original_gaussian']
    auc_df = auc_df[auc_df['XAI_method'] !='cam']
    
    return auc_df

def _get_referee_list_with_criteria(ds):

    filter_df = _get_accuracy_one_ds(ds)
    accuracy_threshold = np.average(filter_df['metrics: acc'])
    if accuracy_threshold > 0.9:
        accuracy_threshold = 0.9
    ref_list = filter_df[filter_df['metrics: acc']> accuracy_threshold]['Referee']
    print(ds, list(ref_list))
    return list(ref_list)


def summarize_result(synthetic_data=False, metric='explanation_power', beautify_display=True,wide_table=True):
    if synthetic_data == True:
        data = SYNTHETIC_DS
        neworder = SYNTHETIC_DATA_ORDER
    else:
        data = UCR_DS
        neworder = UCR_DATA_ORDER
    

    # accuracy_threshold = 0.8 if synthetic_data == True else 0.7
    
    ds = data[0]
    ref_list = _get_referee_list_with_criteria(ds)
    # print(ref_list)
    auc_df = load_xais_comparison_output(ds)
    x = process_auc_df(auc_df,referees=ref_list ,ranking_by_perturbation_method=False, 
                              beautify_display=False)
        
    xais = list(auc_df['XAI_method'].unique())
    colnames = ['dataset']
    for xai in xais:
        colnames.append(xai)
    df = pd.DataFrame(columns=colnames)


    for ds in data[1:]:
        ref_list = _get_referee_list_with_criteria(ds)
        auc_df = load_xais_comparison_output(ds)

        x1 = process_auc_df(auc_df,referees=ref_list ,ranking_by_perturbation_method=False, 
                              beautify_display=False)
        x = x.append(x1)
    
    if not wide_table:
        return x
    table = pd.pivot_table(x, values=metric, 
        index=['dataset'],
        columns=['XAI_method'],
         aggfunc=np.sum)
    table=table.reindex(columns=neworder)

    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=False)
        display(table.style.background_gradient(cmap = cm,axis=0))

    return table

def get_accuracy(synthetic_data=False):
    data = SYNTHETIC_DS if synthetic_data == True else UCR_DS

    ds = data[0]
    x = _get_accuracy_one_ds(ds)
    refs = _get_referees(ds)
    print(refs)
    colnames = ['dataset']
    for ref in refs:
        colnames.append(ref)
    df = pd.DataFrame(columns=colnames)

    for ds in data[1:]:
        x1 = _get_accuracy_one_ds(ds)
        x = x.append(x1)
    
    table = pd.pivot_table(x, values='metrics: acc', 
            index=['dataset'],
            columns=['Referee'],
             aggfunc=np.sum)
    
    return table


def _get_accuracy_one_ds(ds):
    path = glob.glob('./output/acc_%s_*.csv'%(ds))
    assert path, 'No accuracy files found.'
    data = [pd.read_csv(p) for p in path]
    acc_df=pd.concat(data, ignore_index=True, axis=0)
    acc_df = acc_df[acc_df['noise_type'] =='local_mean']
    acc_df = acc_df[acc_df['XAI_method'] =='ridgecv']
    acc_df = acc_df[acc_df['threshold'] == 0]
    
    return acc_df

def _get_referees(ds):
    path = glob.glob('./output/acc_%s_*.csv'%(ds))
    assert path, 'No accuracy files found.'
    data = [pd.read_csv(p) for p in path]
    acc_df=pd.concat(data, ignore_index=True, axis=0)
    
    refs = list(acc_df['Referee'].unique())

    return refs

def xai_average_ranking(ds,referees=None,ranking_by_perturbation_method=False,
                                        display_detail=False):
    colnames = ['dataset','best','worst']
    df = pd.DataFrame(columns=colnames)

    auc_df = load_xais_comparison_output(ds)
    
    x = process_auc_df(auc_df,referees=referees,
                            ranking_by_perturbation_method=ranking_by_perturbation_method, 
                            beautify_display=False)
    x['explanation_power'] = 1-x['scaled_ranking']
    x['ranking'] = x.groupby('dataset')['scaled_ranking'].rank(ascending=True)

    # curr_best = x['XAI_method'][x['scaled_ranking'].idxmin()] 
    # curr_worst = x['XAI_method'][x['scaled_ranking'].idxmax()]
    # df = df.append({'dataset': ds,
    #                'best':curr_best,
    #                 'worst':curr_worst,
    #                }, ignore_index=True)
    # # print(df)
    if display_detail:
        cm = sns.light_palette("green", as_cmap=True,reverse=False)
        display(x.style.background_gradient(cmap = cm,axis=0))
    
    return x

def xai_ranking_by_referee(ds, referee_name):
    auc_df = load_xais_comparison_output(ds)
    x = process_auc_df(auc_df,referees =[referee_name] ,ranking_by_perturbation_method=False,
                             beautify_display=False)
    x['explanation_power'] = 1-x['scaled_ranking']
    x['ranking'] = x.groupby('dataset')['scaled_ranking'].rank(ascending=True)
    print(x)
#   print(x.shape[0])


def get_best_method(ds_list):
    
    colnames = ['dataset','best','worst']
    df = pd.DataFrame(columns=colnames)

    for i,ds in enumerate(ds_list):
        curr_best,curr_worst = xai_average_ranking(ds)
        df = df.append({'dataset': ds,
                       'best':curr_best,
                        'worst':curr_worst,
                       }, ignore_index=True)

    return df

def process_auc_df(auc_df,digit=4,referees=None, beautify_display=True,
    ranking_by_perturbation_method=False,exclude_referees=None,
    print_referees=False):
    # auc_df is the resulted dataframe from CompareExplanation class
    dataset = list(set(auc_df['dataset'].tolist()))
    assert len(dataset) == 1
    dataset=dataset[0]
    xais = set(auc_df['XAI_method'].tolist())
    if referees == None:
        referees = set(auc_df['Referee'].tolist())
    if exclude_referees:
        for item_ in exclude_referees:
            referees.remove(item_)
    if print_referees:
        print('Referees: ',referees)

    pers = set(auc_df['noise_type'].tolist())

    
    col_names=['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc','average_scaled_auc']
    val_df = pd.DataFrame(columns=col_names)
    for ref in referees:
        for noise_type in pers:
            df = auc_df[(auc_df['noise_type']==noise_type) & 
                        (auc_df['Referee']==ref)]
            min_,max_ = df['metrics: explanation_auc'].min(), df['metrics: explanation_auc'].max()
            if min_ != max_:
                df['average_scaled_auc'] = (df['metrics: explanation_auc']-min_)/(max_-min_)
            else:
                df['average_scaled_auc'] = (df['metrics: explanation_auc']-min_)/1.0
            val_df = pd.concat([val_df, df], ignore_index=True, axis=0)
            
    
    val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','noise_type','XAI_method'],
        aggfunc=np.average)
 
    if ranking_by_perturbation_method:
        val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','noise_type','XAI_method'],
        aggfunc=np.average)
    
    else:
        val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','XAI_method'],
        aggfunc=np.average)
    
    val_df = val_df.reset_index()
    col_names=val_df.columns.tolist()
    col_names.append('scaled_ranking')
    ans = pd.DataFrame(columns=col_names)

    
    if ranking_by_perturbation_method:
        for noise_type in pers:
            df = val_df[val_df['noise_type'] == noise_type]
            min_,max_ = df['average_scaled_auc'].min(),df['average_scaled_auc'].max()
            df['scaled_ranking'] = (df['average_scaled_auc']-min_)/(max_-min_)
            ans = pd.concat([ans, df], ignore_index=True, axis=0)

    else:
        df = val_df
        # cal_scaled_ranking(df, ans1)
        min_,max_ = df['average_scaled_auc'].min(),df['average_scaled_auc'].max()
        df['scaled_ranking'] = (df['average_scaled_auc']-min_)/(max_-min_)
        ans = pd.concat([ans, df], ignore_index=True, axis=0)
    
    ans['explanation_power'] = 1-ans['scaled_ranking']
    
    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=False)
        display(ans.style.background_gradient(cmap = cm,axis=0))

    return ans



def summarize_auc(auc_df, beautify_display=True):
    table = pd.pivot_table(auc_df, values='metrics: explanation_auc', 
        index=['dataset','noise_type', 'Referee'],
        columns=['XAI_method'], aggfunc=np.sum)
    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=True)
        display(table.style.background_gradient(cmap = cm,axis=1))

    return table


UCR_DS = ['Car',
'CBF',
'Coffee',
'ECG200',
'ECG5000',
'ECGFiveDays',
'GunPoint',
# 'GunPointAgeSpan',
# 'GunPointMaleVersusFemale',
# 'GunPointOldVersusYoung',
'ItalyPowerDemand',
# 'Meat',
# 'MoteStrain',
'Plane',
'PowerCons',
'SonyAIBORobotSurface1',
'SonyAIBORobotSurface2',
'Strawberry',
'Trace',
'TwoLeadECG',
'TwoPatterns',
# 'Wafer',
# 'WormsTwoClass',
'CMJ',]

SYNTHETIC_DS = ['SmallMiddle_CAR',
 'SmallMiddle_NARMA',
 'SmallMiddle_Harmonic',
 'SmallMiddle_PseudoPeriodic',
 'SmallMiddle_GaussianProcess',
 'RareTime_CAR',
 'RareTime_NARMA',
 'RareTime_Harmonic',
 'RareTime_PseudoPeriodic',
 'RareTime_GaussianProcess']

XAIS_LIST = ['GradientShap','IG','LIME','ROCKET', 'MrSEQL','random','RIDGECV','ROCKET_SHAP','ROCKET_SHAP_NOSCALER','MRSEQL_SHAP']
XAIS_NAMES = ['GradientShap','IntegratedGradient','mrseql-lime','rocket-lime', 'mrseql','random','ridgecv','rocket-shap','rocket-shap-noscaler','mrseql-shap'] 

SYNTHETIC_XAIS_LIST = ['GradientShap','IG','LIME','ROCKET', 'MrSEQL','random','RIDGECV','ROCKET_SHAP', 'GT','MRSEQL_SHAP']
SYNTHETIC_XAIS_NAMES = ['GradientShap','IntegratedGradient','mrseql-lime','rocket-lime', 'mrseql','random','ridgecv','rocket-shap','GT','mrseql-shap'] 

LIME_XAIS_LIST = ['lime','LIME','Lime','ROCKET','ROCKET_SHAP','MRSEQL_SHAP']

SYNTHETIC_DATA_ORDER = ['random2020', 'GS','IG','lime_mrseql','lime_rocket','mrseql-shap','rocket-shap','mrseql','ridgecv','GT']
UCR_DATA_ORDER=['random2020','GS','IG','lime_mrseql','lime_rocket','mrseql-shap','rocket-shap','mrseql','ridgecv']