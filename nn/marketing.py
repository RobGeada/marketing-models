import numpy as np
import os,sys
import time
import matplotlib.pyplot as plt
import nn

cwd = os.getcwd()
#np.random.seed(7)

#===FILE PROCESSING================================
def loadData(trainSize):
	trimData = np.genfromtxt('marketingTrim.csv', delimiter=',')[1:,1:]
	trainScale = (lambda y: np.log(y)/1.),(lambda yPrime: np.exp(yPrime)*1.)

	trimDataX = (trimData-np.mean(trimData,axis=0))/np.std(trimData,axis=0)
	trimDataY = trainScale[0](trimData)

	idx = np.random.rand(len(trimData))<trainSize
	trainX = trimDataX[:,1:][idx]
	trainY = trimDataY[idx][:,0]

	testX = trimDataX[:,1:][~idx]
	testY = trimDataY[~idx][:,0]

	return trainX,trainY,trainScale,testX,testY

#===TESTING SCRIPTS===
#tests a neural net on the given training/testing sets with provided parameters
def singleTest(data,parameters):
	#unpack parameters construct
	hiddenSize,epochs,learningRate = parameters

	#load training,testing data from the specified sets
	trainX,trainY,trainScale,testX,testY = data

	#create network
	print trainX,trainY
	net = nn.Network(inDim=trainX.shape[1],biases=1,hiddenDims=hiddenSize,outDim=1,learningRate=learningRate)

	#train network
	tStart = time.time()
	net.train(trainX,trainY,trainScale,testX,testY,epochs=epochs)

	#display training stats
	print "\n===RESULTS==="
	print "Train time:   {} s".format(time.time()-tStart)

	#make predictions
	tStart = time.time()
	predictionsX = net.predict(testX,trainScale)
	print "Predict time: {} s".format(time.time()-tStart)
	
	#test predictions and display accuracy stats
	net.error(testX,testY,trainScale,verbose=True)
	return np.array(predictionsX),testY


def error(predictions,actual):
	errors = abs((predictions-actual)/actual)
	print "\nMean Error:   {}".format(errors.mean())
	print "Std Dev:      {}".format(np.std(errors))
	print "\nMedian Error: {}".format(np.median(errors))
	print "Max Error:    {}\n".format(errors.max())
	return errors.mean()


#===FIND THE MAXES=======
def nmax(x,n):
	return x.argsort()[-n:][::-1]


#===MAIN=================
if __name__ == "__main__":
	#Specify test parameters
	hiddenSize = [25,15,10]
	trainSize=.95
	epochs = 500
	learningRate = 1e-2
	netParams = (hiddenSize,epochs,learningRate)

	#run test

	data = loadData(trainSize)

	predictions,test = singleTest(data,parameters=netParams)


