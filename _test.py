import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.neural_network import MLPClassifier
import argparse
import torch
import torch.optim as optim
from tqdm import *
from termcolor import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import numpy
import skimage.feature as sk
import skimage
# from package.models import MyModel # you may import your own model in the package
# from package import preprocess_methods # you may import your own preprocess method in the package
#model=torch.load('glcmnet_ori.t7')
#model=torch.load('../glcmnet_ori.t7')

#print(model)

def rgb2gray(rgb):
    #print(rgb.shape)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#4*30*5=600
def get_glcm(I):
    #glcms=torch.FloatTensor(inputs.size()) 
    I=numpy.array(rgb2gray(I.numpy()),dtype='uint8')
    print(I.shape)
    features=[1.00]*600
    glcm=1

    for angle_code in range(1,6):
        for dist in range(1,31):
            if angle_code==1:
                glcm = skimage.feature.greycomatrix(I, [dist], [0], normed=True)
            elif angle_code==2:
                glcm = skimage.feature.greycomatrix(I, [dist], [-dist], normed=True)
            elif angle_code==3:
                glcm = skimage.feature.greycomatrix(I, [0], [-dist], normed=True)
            elif angle_code==4:
                glcm = skimage.feature.greycomatrix(I, [-dist], [-dist], normed=True)
            elif angle_code==5:
                glcm = skimage.feature.greycomatrix(I, [-dist], [0], normed=True)
            #print(glcm.shape)
            fa1= skimage.feature.greycoprops(glcm, 'contrast')[0][0]
            fb1= skimage.feature.greycoprops(glcm, 'energy')[0][0]
            fc1= skimage.feature.greycoprops(glcm, 'homogeneity')[0][0]
            fd1= skimage.feature.greycoprops(glcm, 'correlation')[0][0]
            features[((angle_code-1)*30*4+(dist-1)*4):((angle_code-1)*30*4+dist*4)]=[fa1,fb1,fc1,fd1]
    return torch.from_numpy(numpy.array(features))
    #print(glcms.shape)
    #print(glcm)
    #return torch.from_numpy(glcm)

img=Image.open('data/train/22/train_1.jpg')
#inputs=torch.FloatTensor(64,3,256,256)

grays=get_glcm(torch.from_numpy(numpy.array(img)))
print(grays.shape)
print(grays)
        
'''
img=Image.open('data/train/0/train_28.jpg')
img = img.resize((64,64),Image.ANTIALIAS)
import skimage.feature as sk
##Then I say 

#print(img)
#print(np.array(img).shape)
gray=rgb2gray(np.array(img))
#print(gray.tolist())
#print(np.array(gray,dtype='uint8').tolist())
glcm = sk.greycomatrix(np.array(gray,dtype='uint8'), [1],[0, np.pi/4, np.pi/2, 3*np.pi/4]).transpose(2,3,0,1).reshape(4,256,256)
print(glcm.shape)
'''


#fileObject.close()  
#np.set_printoptions(threshold='nan')  
#print(str(imgnp))
#with open('test.txt','w') as f:
 #   np.savetxt(f, imgnp,fmt='%d')
