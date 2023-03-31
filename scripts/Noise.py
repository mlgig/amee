import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

import random as rd
import matplotlib
import sys
from scipy.interpolate import interp1d
import os

# Noise class to perturb X_series (add noise to important/salient parts of TS only)
# Require: X_series, XAI weight for X_series, noise_type (either global/local mean, 
# global/global gaussian noise, zero replacement)
# Local mean: average of time step value at a particular time step
# Global Mean: average value of time series at any time step


class Noise:
    def __init__(self, X, explanation, noise_type='local_mean'):
        self.X_test = np.squeeze(X)
        self.explanation = explanation
        assert np.squeeze(self.X_test).shape == self.explanation.shape
        self.noise_type = noise_type
        self.X_perturbed_2d = None
        self.X_perturbed_3d = None
        self.noise = None

        self.normalize()

    def normalize(self):
        self.explanation = abs(self.explanation)
        # self.explanation = self.explanation/self.explanation.sum(axis=1, keepdims=1)
        return self.explanation

    def get_noise_sequence(self,seed = 2000):
        np.random.seed(seed)
        rand_matrix = np.random.randn(*self.X_test.shape)

        if self.noise_type == 'zero': 
            noise = np.zeros(self.X_test.shape)
        elif self.noise_type == 'local_mean':
            replace_val = np.mean(self.X_test, axis = 0)
            noise = np.ones(self.X_test.shape)*replace_val
        elif self.noise_type == 'global_mean':
            replace_val = np.mean(self.X_test)
            noise = np.ones(self.X_test.shape)*replace_val
        elif self.noise_type == 'local_gaussian':
            mu = np.mean(self.X_test, axis=0)
            sigma = np.std(self.X_test, axis=0)
            noise = mu + sigma*rand_matrix
        elif self.noise_type == 'global_gaussian':
            mu = np.mean(self.X_test)
            sigma = np.std(self.X_test)
            noise = mu + sigma*rand_matrix
        elif self.noise_type == 'original_gaussian':
            range_ = (np.amax(self.X_test) - np.amin(self.X_test))
            sigma = range_*0.02
            mu = 0
            np.random.seed(2020)
            rand_matrix = np.random.randn(self.X_test.shape[0],self.X_test.shape[-1])
            noise = 1.5*(mu+sigma*rand_matrix)
        else:
            print('Noise type not supported')
            return
        return noise
            
    def add_noise(self, threshold):
        if threshold==0:   
            self.X_perturbed_2d = self.X_test
        else:
            self.noise = self.get_noise_sequence()
            discrim = np.percentile(self.explanation, 100-threshold, axis=1).reshape(-1,1)
            # discrim_area = (self.explanation >= discrim) * 1
            if self.noise_type=='original_gaussian':
                self.X_perturbed_2d = np.where(self.explanation>=discrim, self.X_test+self.noise, self.X_test)
            else:
                self.X_perturbed_2d = np.where(self.explanation>=discrim, self.noise, self.X_test)

        self.X_perturbed_3d = np.expand_dims(self.X_perturbed_2d, 1)

    def visualize(self,idx,threshold,dataset_name):
        plt.figure(figsize = (10,6))
        plt.plot(self.X_perturbed_2d[idx], linewidth=3)
        plt.plot(self.X_test[idx], linewidth=3)
        plt.legend([
            'Noisy',
             'Original'
             ], loc='best')
        plt.title('Sample Signal Pertubed by {} method, threshold = {}, Dataset {}'.format(
            self.noise_type, threshold,dataset_name), fontdict = {'fontsize' : 12})
        # plt.title('Perturbed signal',fontdict = {'fontsize' : 12})