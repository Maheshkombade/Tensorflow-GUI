# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:32:05 2021

@author: prajw
"""
import cv2
import random
import numpy as np

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img

def brightness(img, low = 0.5, high = 3):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def zoom(img, value=0.1):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def channel_shift(img, value = 60):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def convolution(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def blur(img):
    return cv2.blur(img,(5,5))

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)

def median_blur(img):
    return cv2.medianBlur(img, 3)

def dialate(img, kernel = np.ones((3,3))):
    return cv2.dilate(img, kernel)

def erode(img, kernel = np.ones((3,3))):
    return cv2.erode(img, kernel)

def morph_grad(img, kernel = np.ones((3,3))):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def clip(img, th = 3.5):
    mean = np.mean(img)
    std = np.std(img)
    img = np.clip(img, mean-th*std, mean+th*std)
    return img

def trans(img, augs):
    img = img
    for i in augs:
        p = random.random()
        if p>0.5:
            continue
        r = random.random()
        if i == 'horizontal_shift':
            img = horizontal_shift(img, ratio = r)

        if i == 'vertical_shift':
            img = vertical_shift(img, ratio = r)

        if i == 'brightness':
            img = brightness(img)

        if i == 'zoom':

            img = zoom(img, value = r)

        if i == 'channel_shift':
            img = channel_shift(img)

        if i == 'horizontal_flip':

            img = horizontal_flip(img, True)

        if i == 'vertical_flip':

            img = vertical_flip(img, True)

        if i == 'rotation':

            img = rotation(img, angle = r*30)

        if i == 'convolution':
            r = np.random.rand(3,3)
            img = convolution(img, r)

        if i == 'blur':

            img = blur(img)

        if i == 'gaussian':

            img = gaussian_blur(img)

        if i == 'median':

            img = median_blur(img)

        if i == 'dialate':

            img = dialate(img)

        if i == 'erode':

            img = erode(img)

        if i == 'morph':

            img = morph_grad(img)

        if i == 'clip':

            img = clip(img)

    return img
