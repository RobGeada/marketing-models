import numpy as np
import os,sys
import time
import matplotlib.pyplot as plt
import nn
import tensorflow as tf

cwd = os.getcwd()
#np.random.seed(7)

#===FILE PROCESSING================================
def toVect(Y):
	outs = []
	for row in Y:
		out = np.zeros(9)
		out[int(row-1)]=1
		outs.append(out)
	return np.array(outs)


def loadData(trainSize):
	trimData = np.genfromtxt('marketingTrim.csv', delimiter=',')[1:,1:]
	trainScale = (lambda y: y),(lambda yPrime: yPrime)

	trimDataX = (trimData-np.mean(trimData,axis=0))/np.std(trimData,axis=0)
	trimDataY = trainScale[0](trimData)

	idx = np.random.rand(len(trimData))<trainSize
	trainX = trimDataX[:,1:][idx]
	trainY = toVect(trimDataY[idx][:,0])

	testX = trimDataX[:,1:][~idx]
	testY = toVect(trimDataY[~idx][:,0])

	return trainX,trainY,testX,testY

trainX,trainY,testX,testY = loadData(.95)
print trainX.shape,trainY.shape
print testX.shape,testY.shape


#MODEL_PARAMS
inputN = [35,]
hiddenN=[15,15,]
outputN=[9,]
nodes = inputN+hiddenN+outputN
print nodes

#===build model===
sess = tf.InteractiveSession()

#input and output
x = tf.placeholder(tf.float32, shape=[None, 35])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

layers=[x,]

#hidden layers

for i,n in enumerate(nodes[:-1]):
	weight = tf.Variable(tf.random_normal([nodes[i],nodes[i+1]],0,.1))
	bias = tf.Variable(tf.random_normal([nodes[i+1]],0,.1))
	
	if i<len(nodes)-2:	
		print "Layer {} | Weights: {} | Biases: {}".format(i,[nodes[i],nodes[i+1]],nodes[i+1])
		layer_act=tf.add(tf.matmul(layers[i],weight),bias,name='layer_{}_act'.format(i))
		layer=tf.nn.sigmoid(layer_act,name='layer_{}_transfer'.format(i))
		layers.append(layer)
	else:
		keep_prob = tf.placeholder(tf.float32)
		layer_drop = tf.nn.dropout(layers[i], keep_prob)
		print "Layer Output | Weights: {} | Biases: {}".format([nodes[i],nodes[i+1]],nodes[i+1])
		y=tf.add(tf.matmul(layer_drop,weight),bias,name='output_act')

#model parameters
print y_,y
euclidean = tf.sqrt(tf.subtract(y_,y)**2)
loss = tf.reduce_mean(euclidean)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())
#===TRAINING===
print "===TRAINING==="
for i in range(1000):
	train_step.run(feed_dict={x: trainX,y_:trainY,keep_prob:.5})
	if not i%100:
		print "=={}==".format(i)
		print "Train Error: ",loss.eval(feed_dict={x: trainX,y_:trainY,keep_prob:1})
		print "Test Error : ",loss.eval(feed_dict={x: testX,y_:testY,keep_prob:1})
				

#===EVALUATION===
preds = y.eval(feed_dict={x: testX,y_:testY,keep_prob:1})
print preds


