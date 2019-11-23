# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import stuff
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.transform import resize
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# image paths
fpP = "../data/Parasitized/"
fpU = "../data/Uninfected/"

#load images
img_height,img_width = 192,192 

dataP = []
num_imgs = 50
print("Loading {} parasitized cell images".format(num_imgs))
for filename in os.listdir(fpP)[:num_imgs]:
        if '.png' in filename:
            dataP.append(resize(mpimg.imread(fpP+filename), (img_height,img_width,3)))

dataU = []
num_imgs = 50
print("Loading {} unparasitized cell images".format(num_imgs))
for filename in os.listdir(fpU)[:num_imgs]:
    if '.png' in filename:
        dataU.append(resize(mpimg.imread(fpU+filename), (img_height, img_width, 3)))
        
        
dataP = np.array(dataP)
dataU = np.array(dataU)  

# Merge data
p = np.ones(dataP.shape[0])
u = np.zeros(dataU.shape[0])
X = np.concatenate((dataP, dataU))
y = np.concatenate((p, u))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# Normalize image vectors
X_train = X_train/255.
X_test = X_test/255.

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("y_test shape: " + str(y_test.shape))


num_classes = 1
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights= None,
                                            include_top=False,
                                            input_shape= (img_height,img_width,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation= 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam
sgd = SGD(lr=0.5, nesterov=False)
adam = Adam(lr=0.01)
print("Compiling...")
model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Fitting...")
model.fit(X_train, y_train, epochs = 50, batch_size = 64)

preds = model.evaluate(X_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
summary = model.summary()
with open("summary.txt", 'w') as file:
    file.write(str(summary))
print (summary)





























 
