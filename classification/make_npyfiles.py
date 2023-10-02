import os
import numpy as np
import cv2
from PIL import Image

base_dir = os.getcwd()

def num(i):
    i = str(i)
    return '0'*(5-len(i))+i

def combine():
    data=[]
    labels=[]
    test_data =[]
    test_labels =[]
    height = 32
    width = 32
    channels = 3
    classes = 43
    n_inputs = height * width*channels

    p=base_dir+"/User_Custom_Train"
    l=os.listdir(p)
    n_classes=len(l)
    Classes=sorted(l)
    new_classes=Classes[43:n_classes]

    for i in new_classes:
        #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
        path = base_dir+"/train/{}".format(i)
        #print(path)
        if os.path.exists(path):
          Class=os.listdir(path)
          n = len(Class)
          n_test = int(n/2)
          for a in Class:

            image=cv2.imread(path+"/"+a)


            image=np.asarray(image)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
            while(n_test>0):
                test_data.append(np.array(size_image))
                test_labels.append(i)



    data=np.array(data)
    data=data.astype('float32')/255
    test_data = np.array(test_data)
    test_data=data.astype('float32')/255


    np.save(base_dir+'/classification/model/new_data.npy', data) # save
    np.save(base_dir+'/classification/model/new_labels.npy', labels) # save
    np.save(base_dir+'/classification/model/new_test_labels.npy', test_labels)
    np.save(base_dir+'/classification/model/new_test_labels.npy', test_data)
#also make test files
