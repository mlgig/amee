import numpy as np
import os

from .createSimulationDataProcesses import createSimulationDataProcesses
from utils.data import LocalDataLoader



def createDatasets(DatasetsTypes,ImpTimeSteps,ImpFeatures,StartImpTimeSteps,StartImpFeatures,Loc1s,Loc2s,FreezeTypes,isMovings,isPositionals,DataGenerationTypes):

    for i in range(0,len(DatasetsTypes)):
    # for i in [1,4]:

        DatasetsType = DatasetsTypes[i]
        ImpTimeStep=ImpTimeSteps[i]
        ImpFeature=ImpFeatures[i]

        StartImpTimeStep=StartImpTimeSteps[i]
        StartImpFeature=StartImpFeatures[i]

        Loc1=Loc1s[i]
        Loc2=Loc2s[i]

        FreezeType=FreezeTypes[i]
        isMoving=isMovings[i]
        isPositional=isPositionals[i]


        for j in range(len(DataGenerationTypes)):
        # for j in [5,6]:
            if(DataGenerationTypes[j]==None):
                DataName=DatasetsTypes[i]+"_Box"
            else:
                DataName=DatasetsTypes[i]+"_"+DataGenerationTypes[j]
            DataGenerationProcess=DataGenerationTypes[j]

            synth = createSimulationDataProcesses(DatasetsType,ImpTimeStep,ImpFeature,StartImpTimeStep,StartImpFeature,
            	Loc1,Loc2,FreezeType,isMoving,isPositional,DataGenerationProcess)
            synth = synth.makeDataset(DataName=DataName)

    print('FINISHED.')


def write_synth_to_std(DatasetsTypes,DataGenerationTypes):
	for i, DatasetsType in enumerate(DatasetsTypes):
	# for i in [1,4]:
		for j, DataGenerationType in enumerate(DataGenerationTypes):
		# for j in [5,6]:
			if(DataGenerationType==None):
				ds=DatasetsType+"_Box"
			else:
				ds=DatasetsType+'_'+DataGenerationType

			data = LocalDataLoader(dataset=ds)
			X_train,y_train,X_test,y_test = data.get_X_y(synth=True)
			X_train,X_test = np.squeeze(X_train), np.squeeze(X_test)
			y_train,y_test = np.expand_dims(y_train,1), np.expand_dims(y_test,1)

			train = np.hstack((y_train,X_train))
			test = np.hstack((y_test,X_test))

			path = "data/%s" %(ds)
			isExist = os.path.exists(path)
			if not isExist:
				os.makedirs(path)
			train_file =  "data/%s/%s_TRAIN.txt" %(ds,ds)
			test_file = "data/%s/%s_TEST.txt" %(ds,ds)

			np.savetxt(train_file,train)
			np.savetxt(test_file,test)