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

'''
Loading the VGG19 model. This is a pre-trained deep learning model that we can use. We use this to generate features for each image.
This model converts each image to a 1d vector of length 4096. This means that it is somehow compressing the whole image in just 4096 values.
We use these vectors for make further predictions. If we use raw images, that will take a lot of time. These 4096 features have everything that we need 
from the image.
'''
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

'''
Loading data
There should be a folder called 'Cartoons' inside this code directory. Each folder should have 21 sequential frames in it.
A sample folder is attached. Please check it.
'''
folders = []
folderPath = "Cartoons"
for (_,directories,_) in  walk(folderPath):
	folders.extend(directories)
random.shuffle(folders)


#dictionary that will contain features for each folder. This will be a 2d array containing the file number as well as the feature vector.
featuresDict = {}
features = []
labels = []
fCount = 0	#folders count
for folder in folders:
	filenames = []
	start = time.time()
	for(_,_,fnames) in walk(folderPath+"/"+str(folder)):
		filenames.extend(fnames)
	images = []
	label = int(folder.split("-")[1])
	for i in range(0,19,4):

		#image has to be resized and modified slightly to pass it from the model
		image = cv2.imread(folderPath+"/"+str(folder)+"/"+filenames[i])
		image = cv2.resize(image,(224,224))
		images.append(image)

	#all images inside a folder are contained in this 'images' list
	images = np.array(images)

	#predicting features for all images inside a folder. This returns a 5x4096 vector since we are only using 5 frames out of 21.
	predictedFeatures = model.predict(images)
	now = time.time() - start
	print('-- Folder, features shape, time taken, label',folder,predictedFeatures.shape,now,label)

	#dictionary will have folder names as keys and features array as value
	featuresDict[str(folder)] = predictedFeatures
	features.append(predictedFeatures)
	labels.append(label)
	fCount = fCount + 1

	#save a temporary dictonary after 100 folders
	if fCount % 100 == 0:
		np.save('dict-children-'+str(fCount)+'.npy', featuresDict)

#this dictionary will contain all the folders
np.save('dict-children.npy', featuresDict) 