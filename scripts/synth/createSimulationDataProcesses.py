import timesynth as ts
import numpy as np
import sys
import argparse
from .Plotting import *

_NUM_TIMESTEP = 50
_NUM_FEATURES = 50
_SAMPLER = 'irregular'
_HAS_NOISE=True


class createSimulationDataProcesses():
	def __init__(self,DatasetsType,ImpTimeStep,ImpFeature,StartImpTimeStep,StartImpFeature,
		Loc1,Loc2,FreezeType,isMoving,isPositional,DataGenerationProcess):
		self.DatasetsType = DatasetsType
		self.ImpTimeStep = ImpTimeStep
		self.ImpFeature = ImpFeature
		self.StartImpTimeStep = StartImpTimeStep
		self.StartImpFeature = StartImpFeature
		self.Loc1 = Loc1
		self.Loc2 = Loc2
		self.FreezeType = FreezeType
		self.isMoving = isMoving
		self.isPositional = isPositional
		self.DataGenerationProcess = DataGenerationProcess
		self.NumTimeSteps = _NUM_TIMESTEP
		self.NumFeatures = _NUM_FEATURES
		self.Sampler = _SAMPLER
		self.hasNoise = _HAS_NOISE

	def makeDataset(self,DataName,NumTrainingSamples=500,NumTestingSamples=100,plot=True,save=True,datasets_graphs_dir='./plot/',data_dir='./data/synth/'):
		self.DataName = DataName
		if(self.isPositional):
			print("Creating Positional Training Dataset" , self.DataName)
			TrainingDataset, TrainingDataset_MetaData = self.createPositionalDataset(NumTrainingSamples)
			print("Creating Positional Testing Dataset", self.DataName)
			TestingDataset ,TestingDataset_MetaData= self.createPositionalDataset(NumTestingSamples)

		else:

			print("Creating Training Dataset", self.DataName)
			TrainingDataset  , TrainingDataset_MetaData= self.createDataset( NumTrainingSamples)
			print("Creating Testing Dataset", self.DataName)
			TestingDataset ,TestingDataset_MetaData= self.createDataset(NumTestingSamples)

		if(plot==True):
			print("Plotting Samples...")
			if(self.isPositional):
				negIndex=[]
				posIndex=[]

				for i in range(TrainingDataset_MetaData.shape[0]):
					if(TrainingDataset_MetaData[i,0]==1 and  len(posIndex)<2):
						posIndex.append(i)
					elif(TrainingDataset_MetaData[i,0]==0 and  len(negIndex)<2):
						negIndex.append(i)

					if(len(negIndex)==2 and len(posIndex)==2):
						break

				if(self.DataGenerationProcess==None):
					plotExampleBox(TrainingDataset[negIndex[0],:,:],datasets_graphs_dir+self.DataName+'_negtive1' ,flip=True)
					plotExampleBox(TrainingDataset[posIndex[0],:,:],datasets_graphs_dir+self.DataName+'_postive1' ,flip=True)
					
					plotExampleBox(TrainingDataset[negIndex[1],:,:],datasets_graphs_dir+self.DataName+'_negtive2' ,flip=True)
					plotExampleBox(TrainingDataset[posIndex[1],:,:],datasets_graphs_dir+self.DataName+'_postive2' ,flip=True)


				else:
					plotExampleProcesses(TrainingDataset[negIndex[0],:,:],datasets_graphs_dir+self.DataName+'_negtive1')
					plotExampleProcesses(TrainingDataset[posIndex[0],:,:],datasets_graphs_dir+self.DataName+'_postive1',color='b')
					plotExampleProcesses(TrainingDataset[negIndex[1],:,:],datasets_graphs_dir+self.DataName+'_negtive2')
					plotExampleProcesses(TrainingDataset[posIndex[1],:,:],datasets_graphs_dir+self.DataName+'_postive2',color='b')

			else:
				
				negIndex=-1
				posIndex=-1


				for i in range(TrainingDataset_MetaData.shape[0]):
					if(TrainingDataset_MetaData[i,0]==1):
						posIndex=i
						# print(i , TrainingDataset_MetaData[i,:])
					else:
						negIndex=i
						# print(i , TrainingDataset_MetaData[i,:])

					if(negIndex!=-1 and posIndex!=-1):
						break


				plotExampleBox(TrainingDataset[negIndex,:,:],datasets_graphs_dir+self.DataName+'_negtive_heatmap',flip=True)
				plotExampleBox(TrainingDataset[posIndex,:,:],datasets_graphs_dir+self.DataName+'_postive_heatmap',flip=True)

				plotExampleProcesses(TrainingDataset[negIndex,:,:],datasets_graphs_dir+self.DataName+'_negtive_signal')
				plotExampleProcesses(TrainingDataset[posIndex,:,:],datasets_graphs_dir+self.DataName+'_postive_signal',color='b')

		if(save==True):
			print("Saving Datasets...")
			np.save(data_dir+self.DataName+"_TRAIN",TrainingDataset)
			np.save(data_dir+self.DataName+"_TEST",TestingDataset)
			
			np.save(data_dir+self.DataName+"_TRAIN_meta",TrainingDataset_MetaData)
			np.save(data_dir+self.DataName+"_TEST_meta",TestingDataset_MetaData)
			

	def createPositionalDataset(self,NumberOFsamples):
		DataSet = np.zeros((NumberOFsamples,self.NumTimeSteps , self.NumFeatures))
		metaData= np.zeros((NumberOFsamples,5))
		Targets = np.random.randint(-1, 1,NumberOFsamples)

		TargetTS_Ends=np.zeros((NumberOFsamples,))
		TargetFeat_Ends=np.zeros((NumberOFsamples,))

		if (self.FreezeType=="Feature"):

			TargetTS_Starts = np.random.randint(NumTimeSteps-self.ImpTimeStep, size=NumberOFsamples)		
			TargetFeat_Starts=np.zeros((NumberOFsamples,))

			for i in range(NumberOFsamples):
				if(Targets[i]==0):
					Targets[i]=1
					TargetYStart,TargetXStart = TargetTS_Starts[i], self.Loc1
				else:
					TargetYStart,TargetXStart = TargetTS_Starts[i], self.Loc2

				# print(TargetXStart)
				TargetFeat_Starts[i]=TargetXStart

				TargetYEnd,TargetXEnd = TargetYStart+self.ImpTimeStep, TargetXStart+self.ImpFeature

				sample = self.createSample(1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
				if(Targets[i]==-1):
					Targets[i]=0

				TargetTS_Ends[i] = TargetTS_Starts[i]+self.ImpTimeStep
				TargetFeat_Ends[i] = TargetFeat_Starts[i]+self.ImpFeature

				DataSet[i,:,:,]=sample

		else:
			TargetFeat_Starts = np.random.randint( NumFeatures -self.ImpFeature, size=NumberOFsamples)
			TargetTS_Starts=np.zeros((NumberOFsamples,))

			for i in range (NumberOFsamples):
				if(Targets[i]==0):
					Targets[i]=1
					TargetYStart,TargetXStart = Loc1, TargetFeat_Starts[i]
				else:
					TargetYStart,TargetXStart = Loc2, TargetFeat_Starts[i]

				TargetTS_Starts[i]=TargetYStart

				TargetYEnd,TargetXEnd = TargetYStart+self.ImpTimeStep, TargetXStart+self.ImpFeature

				sample = self.createSample(1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
				if(Targets[i]==-1):
					Targets[i]=0

				TargetTS_Ends[i] = TargetTS_Starts[i]+self.ImpTimeStep
				TargetFeat_Ends[i] = TargetFeat_Starts[i]+self.ImpFeature

				DataSet[i,:,:,]=sample

		#Label
		metaData[:,0]=Targets
		#Start important time
		metaData[:,1]=TargetTS_Starts
		#End important time
		metaData[:,2]=TargetTS_Ends
		#Start important feature
		metaData[:,3]=TargetFeat_Starts
		#End important feature
		metaData[:,4]=TargetFeat_Ends

		return DataSet, metaData


	def createDataset(self, NumberOFsamples):
		DataSet = np.zeros((NumberOFsamples, self.NumTimeSteps, self.NumFeatures))
		metaData= np.zeros((NumberOFsamples, 5))
		Targets = np.random.randint(-1, 1, NumberOFsamples)

		TargetTS_Ends=np.zeros((NumberOFsamples,))
		TargetFeat_Ends=np.zeros((NumberOFsamples,))

		if(self.isMoving):
			TargetTS_Starts = np.random.randint(self.NumTimeSteps-self.ImpTimeStep, size=NumberOFsamples)
			TargetFeat_Starts = np.random.randint(self.NumFeatures-self.ImpFeature, size=NumberOFsamples)

		else:
			TargetTS_Starts=np.zeros((NumberOFsamples,))
			TargetFeat_Starts=np.zeros((NumberOFsamples,))

			TargetTS_Starts[:]= self.StartImpTimeStep
			TargetFeat_Starts[:]= self.StartImpFeature

		for i in range (NumberOFsamples):
			if(Targets[i]==0):
				Targets[i]=1

			TargetTS_Ends[i],TargetFeat_Ends[i] = TargetTS_Starts[i]+self.ImpTimeStep, TargetFeat_Starts[i]+self.ImpFeature
			sample = self.createSample(Targets[i],int(TargetTS_Starts[i]),int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]),int(TargetFeat_Ends[i]))

			if(Targets[i]==-1):
				Targets[i]=0

			DataSet[i,:,:,]=sample

		#Label
		metaData[:,0]=Targets
		#Start important time
		metaData[:,1]=TargetTS_Starts
		#End important time
		metaData[:,2]=TargetTS_Ends
		#Start important feature
		metaData[:,3]=TargetFeat_Starts
		#End important feature
		metaData[:,4]=TargetFeat_Ends


		return DataSet,metaData


	def createSample(self,Target,start_ImpTS,end_ImpTS,start_ImpFeat,end_ImpFeat):
		if(self.DataGenerationProcess==None):
			sample=np.random.normal(0,1,[self.NumTimeSteps,self.NumFeatures])
			Features=np.random.normal(Target,1,[self.ImpTimeStep,self.ImpFeature])
			sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]=Features

		else:
			time_sampler = ts.TimeSampler(stop_time=20)
			sample=np.zeros([self.NumTimeSteps,self.NumFeatures])


			if(self.Sampler=="regular"):
				time = time_sampler.sample_regular_time(num_points=self.NumTimeSteps*2, keep_percentage=50)
			else:
				time = time_sampler.sample_irregular_time(num_points=self.NumTimeSteps*2, keep_percentage=50)

			
			for i in range(self.NumFeatures):
				if(self.DataGenerationProcess== "Harmonic"):
					 signal = ts.signals.Sinusoidal(frequency=2)
					
				elif(self.DataGenerationProcess=="GaussianProcess"):
					signal = ts.signals.GaussianProcess(kernel="Matern", nu=3./2)

				elif(self.DataGenerationProcess=="PseudoPeriodic"):
					signal = ts.signals.PseudoPeriodic(frequency=2, freqSD=0.01, ampSD=0.5)

				elif(self.DataGenerationProcess=="AutoRegressive"):
					signal = ts.signals.AutoRegressive(ar_param=[0.9])

				elif(self.DataGenerationProcess=="CAR"):
					signal = ts.signals.CAR(ar_param=0.9, sigma=0.01)

				elif(self.DataGenerationProcess=="NARMA"):
					signal = ts.signals.NARMA(order=10)

				if(self.hasNoise):
		 			noise= ts.noise.GaussianNoise(std=0.3)
				 	timeseries = ts.TimeSeries(signal, noise_generator=noise)
				else:
				 	timeseries = ts.TimeSeries(signal)

				feature, signals, errors = timeseries.sample(time)
				sample[:,i]= feature



			sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]=sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]+Target
		return sample