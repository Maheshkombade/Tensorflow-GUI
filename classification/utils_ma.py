# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:50:29 2021

@author: prajw
"""

# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import numpy as np

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adam

import innvestigate
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis

import innvestigate.utils
import innvestigate.utils.tests
import innvestigate.utils.tests.networks
import os

###############################################################################
# Data Preprocessing Utility
###############################################################################

base_dir=os.getcwd()
def fetch_data():
    data=[]
    labels=[]

    labels = np.load(base_dir+'/classification/model/labels.npy')
    data = np.load(base_dir+'/classification/model/data.npy')

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


    return X_train, y_train, X_val, y_val


def create_preprocessing_f(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """

    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(
                len(input_range)))
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(
                input_range))

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= (b-a)
        X *= (d-c)
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= (d-c)
        X *= (b-a)
        X += a
        return X

    return preprocessing, revert_preprocessing


############################
# Model Utility
############################


def create_model(modelname, **kwargs):
    channels_first = K.image_data_format() == "channels_first"
    num_classes = 10

    if channels_first:
        input_shape = (None, 1, 28, 28)
    else:
        input_shape = (None, 28, 28, 1)

    # load PreTrained models
    if modelname in innvestigate.applications.mnist.__all__:
        model_init_fxn = getattr(innvestigate.applications.mnist, modelname)
        model_wo_sm, model_w_sm = model_init_fxn(input_shape[1:])

    elif modelname in innvestigate.utils.tests.networks.base.__all__:
        network_init_fxn = getattr(innvestigate.utils.tests.networks.base,
                                   modelname)
        network = network_init_fxn(input_shape,
                                   num_classes,
                                   **kwargs)
        model_wo_sm = Model(inputs=network["in"], outputs=network["out"])
        model_w_sm = Model(inputs=network["in"], outputs=network["sm_out"])
    else:
        raise ValueError("Invalid model name {}".format(modelname))

    return model_w_sm


def train_model(model, data, batch_size=128, epochs=20):
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


############################
# Post Processing Utility
############################


def postprocess(X):
    X = X.copy()
    X = iutils.postprocess_images(X)
    return X


def bk_proj(X):
    return ivis.graymap(X)


def heatmap(X):
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_positive_only=True)
