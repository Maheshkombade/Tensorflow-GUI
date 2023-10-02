# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:29:14 2021

@author: prajw
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:04:50 2021

@author: prajw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
import logging
import random
from augmentations import *
import tqdm
tf.get_logger().setLevel(logging.ERROR)

# Reading the input images and putting them into a numpy array
data=[]
labels=[]

height = 32
width = 32
channels = 3
classes = 43
n_inputs = height * width*channels

#%%
'''
augs = ['horizontal_shift',
        'vertical_shift',
        'brightness',
        'zoom',
        'channel_shift',
        'horizontal_fl0ip',
        'vertical_flip',
        'rotation',
        'convolution',
        'blur',
        'gaussian',
        'median',
        'dialate',
        'erode',
        'morph',
        'clip']

'''
augs = ['horizontal_shift',
        'brightness',
        'zoom']

def disp(im1,im2,n,save=False):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(im1)
    axs[0].set_title('Original')
    axs[1].imshow(im2)
    axs[1].set_title('Augmented')
    fig.suptitle('Class {}'.format(n))P
    plt.show()
    plt.savefig('class_{}_aug.jpg'.format(n))

arr = [0]
for i in range(1,len(labels)):
    if labels[i]!=labels[i-1]:
        arr.append(i)
arr.append(39209)

rands = []

for i in range(len(arr)-1):
    a=random.randint(arr[i], arr[i+1])
    b=random.randint(arr[i], arr[i+1])
    rands.append(min(a,b))
    rands.append(max(a,b))



#%%
'''
def num(i):
    i = str(i)
    return '0'*(5-len(i))+i


for i in range(classes) :
    path = "data/Final_Training/Images/{0}/".format(num(i))
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

'''
labels = np.load('labels.npy')
data_orig = np.load('data.npy')
data = np.load('data.npy')
cl_num = 0
for i in tqdm.tqdm(range(len(data))):
    try:
        data[i] = trans(data[i], augs)
    except:
        pass
    if i in rands:
        disp(data_orig[i], data[i], cl_num//2)
        cl_num+=1

#%%
#Randomize the order of the input images
Cells=np.array(data)
labels=np.array(labels)
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

#Spliting the images into train and validation sets
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
'''
lr_schedule = ExponentialDecay(
initial_learning_rate=1e-2,
decay_steps=10000,
decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
'''
#Compilation of the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

'''
LR = ReduceLROnPlateau(monitor="val_loss",
    factor=0.1,
    patience=3,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model--1-ii.h5', verbose=1, save_best_only=True)
'''
model.summary()

epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))

#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

y_test=pd.read_csv("data/GT-final_test.csv", delimiter=';',)
labels=y_test['Filename'].values
y_test=y_test['ClassId'].values
np.save('y_test.npy', y_test)

data=[]


for f in tqdm.tqdm(labels):
    image=cv2.imread('data/test/'+f)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))


X_test=np.array(data)
X_test = X_test.astype('float32')/255
np.save('X_test.npy', X_test)

pred = np.argmax(model.predict(X_test), axis = 1)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, pred)
m = confusion_matrix(y_test, pred)
plt.matshow(m)
plt.title('accuracy = {}'.format(accuracy_score(y_test, pred)))
plt.show()

model.save_weights('ii_aug_no_dat.h5')

