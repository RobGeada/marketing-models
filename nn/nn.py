import numpy as np
import string
import os,sys
import time

cwd = os.getcwd()
#np.random.seed(7)

#====ACTIVATION FUNCTIONS===================
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x)*(1-sigmoid(x))

def softPlus(x):
	return np.maximum(x,np.zeros(x.size))

def softPlusPrime(x):
	x[x<0]=0
	x[x>=0]=1
	return(x)

#===LAYER DEFINITIONS=====================
class Layer:
	def __init__(self,name,size,transfers,biases=0):
		self.size = size+biases
		self.biases = biases
		self.name = name

		self.sums = np.zeros((self.size))
		self.xs = np.zeros((self.size))
		
		self.antWeights = np.array(1)
		self.ant = None
		self.connWeights= np.array(1)
		self.conn = None

		self.deltas = np.zeros((self.size))
		self.transfer = transfers[0]
		self.transferPrime =  transfers[1]

	#fully link current layer to next
	def pointsTo(self,layer): 
		weightRange = 1/np.sqrt(self.size)
		
		#connect this layer to next layer
		self.connWeights = weightRange*np.random.randn(self.size,layer.size)
		self.conn = layer

		#connect next layer to this layer
		layer.antWeights = np.transpose(self.connWeights)
		layer.ant = self

	#assign outputs to nodes (creates input layer)
	def setValues(self,inputs):
		self.xs = np.append(inputs,np.ones(self.biases))

	#calculate all node output within layer (for computing output)
	def calculate(self):
		if self.ant!=None:
			self.sums = np.dot(self.antWeights,self.ant.xs)
			self.xs = self.transfer(self.sums)

	#functions to display layer params for testing and debugging
	def display(self,verbose=False):
		if verbose:
			print "==========LAYER {}==========".format(self.name)
			for i,node in enumerate(self.xs):
				print "Node {} | Sends: {} | Receives: {} | Output: {}".format(i,len(self.connWeights[i]),len(self.antWeights[i]),node)
		else:
			print "{} | Size: {} | Connections: {} | Ants: {}".format(self.name,self.size,self.connWeights.shape,self.antWeights.shape)

#===NETWORK HELPERS==========================
def layerNamer(n):
	name = ""
	nameSize = (n - (n%26))/26
	alphabet = string.ascii_lowercase
	for i in range(0,nameSize+1):
		name+=alphabet[n%26]
	return name

#===NETWORK CLASS============================
class Network:
	def __init__(self,inDim,biases,hiddenDims,outDim,learningRate):
		#initialize network
		self.hiddens = len(hiddenDims)
		self.hiddenLayers = []
		self.learningRate = learningRate
		self.input = Layer(name="Input",size=inDim,biases=biases,transfers=[sigmoid,sigmoidPrime])
		for i,layerDim in enumerate(hiddenDims):
			self.hiddenLayers.append(Layer(name=layerNamer(i),size=layerDim,biases=1,transfers=[sigmoid,sigmoidPrime]))
		self.output = Layer(name="Output",size=outDim,transfers=[sigmoid,sigmoidPrime])

		#link network
		self.input.pointsTo(self.hiddenLayers[0])
		for h,hiddenLayer in enumerate(self.hiddenLayers):
			if h<self.hiddens-1:
				self.hiddenLayers[h].pointsTo(self.hiddenLayers[h+1])
			else:
				self.hiddenLayers[h].pointsTo(self.output)

	#given current network inputs, produce outputs
	def calculate(self):
		self.input.calculate()
		for layer in reversed(self.hiddenLayers):
			layer.calculate()
		self.output.calculate()

	#train on single example
	def trainExample(self,input,trainOut):
		self.input.setValues(input)
		self.calculate()
		results = self.output.xs
		#print results,trainOut

		trainLayers = self.hiddenLayers[:]
		trainLayers.append(self.output)

		#backpropagate layers
		for layer in reversed(trainLayers):
			#print layer.name
			if "Output" in layer.name:
				layer.deltas = (layer.xs - trainOut)*layer.transferPrime(layer.sums)
			else:
				layer.deltas = layer.transferPrime(layer.sums)*np.dot(layer.connWeights,layer.conn.deltas)			
			layer.ant.connWeights = layer.ant.connWeights - self.learningRate*layer.ant.xs[:,None]*layer.deltas
			layer.antWeights = np.transpose(layer.ant.connWeights)

	#train over entire dataset
	def train(self,trainX,trainY,scaleFunc,testX,testY,epochs,verbose=True):
		lastError = 10
		holdoutError = 1
		epoch = 0
		errors = []
		if verbose:
			print "Training network..."
		while epoch<epochs:#(lastError*1.05)>(holdoutError):
			lastError = holdoutError
			if epoch%40==0:
				print "adjusting"
				self.learningRate *= 1
			if verbose:
				print "===EPOCH {} LR {}===".format(epoch,self.learningRate)
			for i,x in enumerate(trainX):
				#print "training point {}".format(i)
				#normX,normY = normalize(x,trainY[i])
				normX,normY = (x,trainY[i])
				self.trainExample(normX,normY)
			

			#trainError= self.error(trainX,trainY,scaleFunc,verbose=True)
			holdoutError = self.error(testX,testY,scaleFunc)
			errors.append(holdoutError)
			if verbose:
				print "Holdout error: {}".format(holdoutError)
			epoch+=1
		return errors

	#compute prediction error on given dataset 
	def error(self,testX,testY,trainScale,verbose=False):
		errors = np.zeros(len(testY))
		corrects = 0
		dists = np.zeros(len(testY))
		for i,x in enumerate(testX):
			#x,y = normalize(x,testY[i])
			y = testY[i]
			self.input.setValues(x)
			self.calculate()
			results = trainScale[1](self.output.xs)
			if np.argmax(results)==np.argmax(y):
				corrects+=1
			errors[i]= np.sqrt(np.sum((results-y)**2))

			dists[i]=abs(np.argmax(results)-np.argmax(y))
		print "Correct: {}/{}".format(corrects,len(testY))
		print "MAD: {}".format(dists.mean())
		if not verbose:
			return errors.mean()
		else:
			#print "\nMean Error:   {}%".format(errors.mean()*100)
			#print "Std Dev:      {}%".format(np.std(errors)*100)
			#print "\nMedian Error: {}%".format(np.median(errors)*100)
			#print "Max Error:    {}%\n".format(errors.max()*100)
			return errors.mean()

	#create predictions
	def predict(self,xs,trainScale):
		predictions = []
		for i,x in enumerate(xs):
			self.input.setValues(x)
			self.calculate()
			results = self.output.xs
			predictions.append(trainScale[1](results[0]))
		return predictions

	#debugging
	def display(self,verbose=False):
		self.input.display(verbose)
		for h in self.hiddenLayers:
			h.display(verbose)
		self.output.display(verbose)
