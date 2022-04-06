import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from pyts.transformation import WEASEL
from .Noise import Noise

class Evaluate():
	def __init__(self,X_train,y_train,X_test,y_test,explanation,
		referee='MrSEQLClassifier',step=10, noise_type='zero'):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.explanation = explanation
		self.referee = referee
		self.step = step
		self.noise_type = noise_type
		self.model = self.train_referee()
		self.result = self.evaluate_stepwise()
		self.explanation_auc = self.get_explanation_auc()
	def train_referee(self,):
		if self.referee == 'MrSEQLClassifier':
			model = MrSEQLClassifier(seql_mode="fs")
			model.fit(self.X_train,self.y_train)
		elif self.referee == 'minirocket':
			self.transformer = MiniRocket()  # by default, MiniRocket uses ~10,000 kernels
			self.transformer.fit(self.X_train)
			X_train_transform = self.transformer.transform(self.X_train)
			# self.X_test_transform = self.transformer.transform(self.X_test)

			model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
			model.fit(X_train_transform, self.y_train)

		elif self.referee == 'weasel':
			ts_len = self.X_train.shape[-1]
			window_size= np.arange(5,ts_len, 1)
			n_class = len(set(np.unique(self.y_test)))
			if n_class == 2: 
				self.transformer = WEASEL(sparse=False,window_sizes=window_size, n_bins=2)
			else: 
				self.transformer = WEASEL(sparse=False,window_sizes=window_size)

			X_train_2d = np.squeeze(self.X_train)
			self.transformer.fit(X_train_2d,self.y_train)
			X_train_transform = self.transformer.transform(X_train_2d)
			model = LogisticRegression(random_state=2)
			model.fit(X_train_transform, self.y_train)
		return model

	def get_accuracy(self,X_test_perturbed):
		if self.referee in ['minirocket','rocket','weasel']:
			if self.referee == 'weasel':
				X_test_perturbed = np.squeeze(X_test_perturbed)
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

	def record_result(self,dataset_name='Coffee',explanation_name='MrSeql-SM', 
		existing_df=False, df=None):
		if existing_df == False:
			col_names=['dataset','noise_type', 'XAI_method', 'Referee', 'threshold','metrics: acc']
			df = pd.DataFrame(columns=col_names)
		else: df=pd.DataFrame(df)

		for threshold,acc in self.result.items():
			df = df.append({'dataset': dataset_name,
				'noise_type': self.noise_type,
				'XAI_method': explanation_name,
				'Referee': self.referee,
				'threshold': threshold,
				'metrics: acc': acc}, ignore_index=True)
		return df

	def record_auc(self,dataset_name='Coffee',explanation_name='MrSeql-SM',
		existing_df=False, auc_df=None):
		if existing_df == False:
			col_names = ['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc']
			auc_df = pd.DataFrame(columns=col_names)
		else: auc_df =pd.DataFrame(auc_df)

		auc_df = auc_df.append({'dataset': dataset_name,
				'noise_type': self.noise_type,
				'XAI_method': explanation_name,
				'Referee': self.referee,
				'metrics: explanation_auc': self.explanation_auc}, ignore_index=True)
		return auc_df


		


