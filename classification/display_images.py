import cv2
import os
import random
from PIL import Image
from .augmentations import *

base_dir = os.getcwd()

def display(condition,augs):   #give augs list to it
    path=base_dir+"/User_Custom_Train/"
    sign_class=os.listdir(path)
    n_classes=len(sign_class)
    sign_class1=sorted(sign_class)
    if(condition=="second"): #add augmentations and then merge with the training data



        s=sign_class1[43:n_classes]

        for i in s:
            files=os.listdir(path+i)
            im_random=random.choice(files)
            image=Image.open(path+i+"/"+im_random)
            im=cv2.imread(path+i+"/"+im_random)
            os.chdir(base_dir+"/classification/static/augmented_images/new_classes/original")
            n=i+".jpg"
            cv2.imwrite(n,im)
            a=trans(im,augs)
            os.chdir(base_dir+"/classification/static/augmented_images/new_classes/augmented")
            name="aug"+i+".jpg"
            cv2.imwrite(name,a)

    elif(condition=="first"):    #Merge and then add augmentations


        s=sign_class1


        for i in s:
            files=os.listdir(path+i)
            im_random=random.choice(files)
            image=Image.open(path+i+"/"+im_random)
            im=cv2.imread(path+i+"/"+im_random)
            print("dipanshu", os.getcwd())
            os.chdir(base_dir+"/classification/static/augmented_images/all_classes/original")
            n=i+".jpg"
            cv2.imwrite(n,im)
            a=trans(im,augs)
            os.chdir(base_dir+"/classification/static/augmented_images/all_classes/augmented")

            name="aug"+i+".jpg"
            cv2.imwrite(name,a)



    elif(condition == "third"):
            s=sign_class1[0:43]
            for i in s:
                files=os.listdir(path+i)
                im_random=random.choice(files)
                image=Image.open(path+i+"/"+im_random)
                im=cv2.imread(path+i+"/"+im_random)
                os.chdir(base_dir+"/classification/static/augmented_images/Orig_classes/original")
                n=i+".jpg"
                cv2.imwrite(n,im)
                a=trans(im,augs)
                os.chdir(base_dir+"/classification/static/augmented_images/Orig_classes/augmented")
                name="aug"+i+".jpg"
                cv2.imwrite(name,a)


