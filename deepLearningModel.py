from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input
import cv2
import numpy as np
import os
import sys
from os import walk
import keras
from keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,Input,Reshape,GRU,LSTM
from keras.models import Sequential
from keras import optimizers
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

DICTIONARY_FILE = str(sys.argv[1])

"""
Example usage: python deepLearningModel.py dictionaryFile.npy

This function creates our deep learning model and trains it on our dataset that is present inside the dictionary
"""

def getSingleLabels(predictions,y_test):
	"""
	Helper function
	"""
	predictionsSingle = []
	actualSingle = []
	for p in range(0,len(predictions)):
		pred = predictions[p]
		actual = y_test[p]
		if pred[0] > 0.5:
			predictionsSingle.append(0)
		elif pred[1] >= 0.5:
			predictionsSingle.append(1)
		if actual[0] == 1:
			actualSingle.append(0)
		elif actual[1] == 1:
			actualSingle.append(1)
	return predictionsSingle,actualSingle

#load dictionary file
featuresDict = np.load(DICTIONARY_FILE).item()

features = []
labels = []
folders = []
for f in featuresDict:
	#f is the key and featuresDict[f] is the feature vector
	label = int(f.split("-")[1])
	featureVec = featuresDict[f]
	features.append(featureVec)
	labels.append(label)
	folders.append(f)

features = np.array(features)
print('-- Label distribution',list(labels).count(0),list(labels).count(1))
labels = np.array(keras.utils.to_categorical(labels, num_classes=2))
print('-- Input shape is',features.shape,labels.shape)

#splitting the data into 80-20, 80% is the training data and 20% is the testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)

#creating the architecture
model = Sequential()
model.add(Dense(512,input_shape=features.shape[1:],kernel_initializer='random_uniform'))
model.add(Dense(128,kernel_initializer='random_uniform'))
model.add(Dense(128,kernel_initializer='random_uniform'))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))

#lr is the learning rate
optimizer = optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
print(model.summary())

print('***** Staring model training *****')
EPOCHS = 3
for j in range(0,EPOCHS):
	print('-- Iteration',j)
	model.fit(X_train, y_train,batch_size=1,epochs=1,verbose=2,shuffle=True)
	scoreTest = model.evaluate(X_test,y_test)[1]
	scoreTrain = model.evaluate(X_train,y_train)[1]
	print('\nTraining and testing scores are',scoreTrain,scoreTest)
	

