import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from weasel.classification.dictionary_based import WEASEL_V2

import torch
from torch.utils.data import DataLoader, TensorDataset
from .Noise import Noise
from utils.data import LocalDataLoader


class Evaluate():
	def __init__(self,datapath, dataset, explanation,referee='MrSEQLClassifier', step=10, 
		noise_type='zero',model=None,transformer=None, evaluation_on_subset=False, 
		eval_size=0.2):
		self.dataset = dataset
		data = LocalDataLoader(datapath,self.dataset)
		self.X_train,self.y_train,self.X_test,self.y_test = data.get_X_y()

		_,_,_,self.y_test_onehot = data.get_X_y(onehot_label=True)
		if evaluation_on_subset:
			_,self.X_test,_,self.y_test = train_test_split(self.X_test, self.y_test, test_size=eval_size, random_state=9000)
			encoder = OneHotEncoder(categories='auto', sparse=False)
			encoder.fit(np.expand_dims(self.y_train, axis=-1))
			self.y_test_onehot = encoder.transform(np.expand_dims(self.y_test, axis=-1))
        	
		self.explanation = explanation
		self.referee = referee
		self.step = step
		self.noise_type = noise_type
		if model == None:
			model,transformer = train_referee(self.X_train,self.y_train,self.referee,self.dataset)
		self.model = model
		self.transformer = transformer
		self.result = self.evaluate_stepwise()
		self.explanation_auc = self.get_explanation_auc()
	
	def get_accuracy(self,X_test_perturbed):
		if self.referee == 'resnet':
			# create a new test_loader for perturbed X_test
			test_dataset = TensorDataset(torch.from_numpy(X_test_perturbed).float(),
				torch.from_numpy(self.y_test_onehot))
			test_loader = DataLoader(test_dataset, batch_size=64)

			self.model.eval()
			true_list,pred_list = [],[]

			for x,y in test_loader:
				with torch.no_grad():
					true_list.append(y.detach().numpy())
					preds = self.model(x)
					preds=torch.softmax(preds,dim=-1)
					pred_list.append(preds.detach().numpy())
			true_np,preds_np = np.concatenate(true_list), np.concatenate(pred_list)

			true_np = np.argmax(true_np,axis=-1)
			preds_np= np.argmax(preds_np,axis=-1)
			acc = metrics.accuracy_score(true_np,preds_np)


		elif self.referee in ['minirocket','rocket']:
			# if self.referee == 'weasel':
			# 	X_test_perturbed = np.squeeze(X_test_perturbed)
			X_test_perturbed_transform = self.transformer.transform(X_test_perturbed)
			acc=self.model.score(X_test_perturbed_transform,self.y_test)
		else:
			predicted = self.model.predict(X_test_perturbed)
			acc = metrics.accuracy_score(self.y_test, predicted)

		return acc

	def evaluate_stepwise(self,):
		noise = Noise(X=self.X_test,explanation=self.explanation,noise_type=self.noise_type)
		result = dict()

		for threshold in range(0,101,self.step):
			#get perturbed X_test
			noise.add_noise(threshold=threshold)
			X_perturbed = noise.X_perturbed_3d
			acc_perturbed = self.get_accuracy(X_perturbed)
			result[threshold]=acc_perturbed
		return result

	def get_explanation_auc(self,):
		acc = [val for val in self.result.values()]
		steps = np.arange(0,1.1, 0.1)
		exp_auc = metrics.auc(steps, acc)
		return exp_auc

	# def get_explanation_crossentropy(self,):

	def record_result(self,explanation_name='MrSeql-SM', 
		existing_df=False, df=None):
		if existing_df == False:
			col_names=['dataset','noise_type', 'XAI_method', 'Referee', 'threshold','metrics: acc']
			df = pd.DataFrame(columns=col_names)
		else: df=pd.DataFrame(df)

		for threshold,acc in self.result.items():
			df = df.append({'dataset': self.dataset,
				'noise_type': self.noise_type,
				'XAI_method': explanation_name,
				'Referee': self.referee,
				'threshold': threshold,
				'metrics: acc': acc}, ignore_index=True)
		return df

	def record_auc(self,explanation_name='MrSeql-SM',
		existing_df=False, auc_df=None):
		if existing_df == False:
			col_names = ['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc']
			auc_df = pd.DataFrame(columns=col_names)
		else: auc_df =pd.DataFrame(auc_df)

		auc_df = auc_df.append({'dataset': self.dataset,
				'noise_type': self.noise_type,
				'XAI_method': explanation_name,
				'Referee': self.referee,
				'metrics: explanation_auc': self.explanation_auc}, ignore_index=True)
		return auc_df

########################################################################
def train_referee(X_train,y_train,referee,dataset):
	transformer = None
	if referee == 'MrSEQLClassifier':
		model = MrSEQLClassifier(seql_mode="fs")
		model.fit(X_train,y_train)

	elif referee == 'knn':
		model = KNeighborsTimeSeriesClassifier(distance="euclidean")
		model.fit(X_train,y_train)

	elif referee == 'rocket':
		transformer = Rocket()  
		transformer.fit(X_train)
		X_train_transform = transformer.transform(X_train)
		model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
		model.fit(X_train_transform, y_train)

		
	elif referee == 'minirocket':
		transformer = MiniRocket()  # by default, MiniRocket uses ~10,000 kernels
		transformer.fit(X_train)
		X_train_transform = transformer.transform(X_train)
		# self.X_test_transform = self.transformer.transform(self.X_test)

		model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
		model.fit(X_train_transform, y_train)

	elif referee == 'weasel':
		model = WEASEL_V2(random_state=1379, n_jobs=4)
		model.fit(X_train,y_train)


	elif referee == 'resnet':
		model_path = './model/%s_best.pkl' %dataset
		model = torch.load(model_path)

	return model, transformer

	


