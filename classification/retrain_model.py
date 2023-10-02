# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
import logging
import random
from .augmentations import *
import tqdm
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import plotly.graph_objects as go
import plotly.express as px
from .make_npyfiles import *


from tensorflow.keras.utils import to_categorical

base_dir = os.getcwd()

def disp(im1,im2,n,save=False):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(im1)
    axs[0].set_title('Original')
    axs[1].imshow(im2)
    axs[1].set_title('Augmented')
    fig.suptitle('Class {}'.format(n))
    plt.show()
    plt.savefig('class_{}_aug.jpg'.format(n))

def retrain(condition,augs,classes):

    #Original data is in a data.npy file and the new images are in train folder
    #condition (first) - retrain data.npy without any augments
    #condition (second) - retrain data.npy with augments
    #condition (Third)- apply augments on combined classes and then retrain
    #condition (Fourth)- apply augments on new classes and then combine and retrain

    # Reading the input images and putting them into a numpy array
    data=[]
    labels=[]

    height = 32
    width = 32
    channels = 3
    n_inputs = height * width*channels

    #%%
    '''
    augs = ['horizontal_shift',
            'vertical_shift',
            'brightness',
            'zoom',
            'channel_shift',
            'horizontal_flip',
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




    arr = [0]
    for i in range(1,len(labels)):
        if labels[i] != labels[i-1]:
            arr.append(i)
    arr.append(39209)

    rands = []

    for i in range(len(arr)-1):
        a=random.randint(arr[i], arr[i+1])
        b=random.randint(arr[i], arr[i+1])
        rands.append(min(a,b))
        rands.append(max(a,b))


    cl_num = 0
    labels = np.load(base_dir+'/classification/model/labels.npy', allow_pickle=True)
    data= np.load(base_dir+'/classification/model/data.npy', allow_pickle=True)
    test_data=np.load(base_dir+"/classification/model/X_test.npy")
    test_labels=np.load(base_dir+"/classification/model/y_test.npy")
    data2=[]
    if(condition=="first"): #first condition
        pass

    elif(condition=="second"): #second condition
        for i in tqdm.tqdm(range(len(data))):
            try:
                temp= trans(data[i], augs)
                data2.append(temp)
            except:
                pass
        data=np.concatenate((data,data2))
        labels=np.concatenate((labels,labels))

    elif(condition == "third"):
        combine()
        new_labels=np.load(base_dir+"/classification/model/new_labels.npy")
        new_data = np.load(base_dir+"/classification/model/new_data.npy")
        data=np.concatenate((data,new_data),axis=0)
        labels= np.concatenate((labels,new_labels),axis=0)
        new_test_data=np.load(base_dir+"/classification/model/new_test_data.npy")
        new_test_labels=np.load(base_dir+"/classification/model/new_test_labels.npy")
        X_test = np.concatenate((X_test,new_test_data))
        y_test = np.concatenate((y_test,new_test_labels))
        for i in tqdm.tqdm(range(len(data))):
            try:
                temp= trans(data[i], augs)
                data2.append(temp)
            except:
                pass
        data=np.concatenate((data,data2))
        labels=np.concatenate((labels,labels))
    elif(condition == "forth"):
        combine()

        new_labels=np.load(base_dir+"/classification/model/new_labels.npy")
        new_data = np.load(base_dir+"/classification/model/new_data.npy")
        new_test_data=np.load(base_dir+"/classification/model/new_test_data.npy")
        new_test_labels=np.load(base_dir+"/classification/model/new_test_labels.npy")
        X_test = np.concatenate((X_test,new_test_data))
        y_test = np.concatenate((y_test,new_test_labels))

        for i in tqdm.tqdm(range(len(new_data))):
            try:

                temp= trans(new_data[i], augs)
                data2.append(temp)
            except:
                pass
        new_data=np.concatenate((new_data,data2))
        data=np.concatenate((data,new_data),axis=0)
        labels= np.concatenate((labels,new_labels),axis=0)
        labels= np.concatenate((labels,new_labels),axis=0)



    #%%
    #Randomize the order of the input images
    Cells=np.array(data)
    labels=np.array(labels)
    s=np.arange(Cells.shape[0])
    np.random.seed(classes)
    np.random.shuffle(s)
    Cells=Cells[s]
    labels=labels[s]

    #Spliting the images into train and validation sets
    (X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
    X_train = X_train.astype('float32')/255
    X_val = X_val.astype('float32')/255
    (y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

    #Using one hote encoding for the train and validation labels

    y_train = to_categorical(y_train, classes)
    y_val = to_categorical(y_val, classes)



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
    model.add(Dense(classes, activation='softmax'))
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

    epochs = 30
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
    validation_data=(X_val, y_val))








    pred = np.argmax(model.predict(test_data), axis = 1)



    # plt.matshow(m)
    # plt.title('accuracy = {}'.format(accuracy_score(y_test, pred)))
    # plt.show()
    # model_json = model.to_json()
    # with open(base_dir+'/classification/model/ii.json', "w") as json_file:
    #     json_file.write(model_json)

    model.save_weights(base_dir+'/classification/new_model/new_model_weights.h5')   #give correct path to save the model

    model.save(base_dir+"/classification/new_model/new_model.h5")


    np.save(base_dir+'/classification/new_model/train_loss.npy',np.array(history.history['loss']))
    np.save(base_dir+'/classification/new_model/val_loss.npy',np.array(history.history['val_loss']))
    accu_score=accuracy_score(test_labels,pred)
    m = confusion_matrix(test_labels, pred)


    return history.history.get('acc'),history.history['val_acc'],history.history['loss'],history.history['val_loss'],m,accu_score







