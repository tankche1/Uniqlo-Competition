# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:51:07 2017

@author: n.aoi
"""

import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from collections import OrderedDict
from tqdm import *
from termcolor import *
from torch.autograd import Variable

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import affine_transforms
# from package import preprocess_methods # if you used some preprocess method in training phase, you may want to apply it in test phase.

PATH_TO_TEST_IMAGES = os.path.join('data', 'test')
PATH_TO_SUBMIT_FILE = 'submitfc.csv'

BatchSize=64


testdir='data/test1/0'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#print(os.listdir(traindir))

# hash2 map classifier to correct number
hash2=['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '3', '4', '5', '6', '7', '8', '9']

AA=datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


AffineMethod=affine_transforms.Affine(rotation_range=10,translation_range=[0.1,0.1],shear_range=0.1,zoom_range=[0.9,1.1])
'''
class Transpose(object):
    def __init__(self,type=1):
        self.type=type
    def __call__(self,x):
        x=x.numpy()
        #print(x.shape,self.type)
        if self.type==1:
            return torch.from_numpy(x.transpose(1,2,0))
        else:
            return torch.from_numpy(x.transpose(2,0,1))
'''

test_loader0 = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            #transforms.Scale(224),
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BatchSize, shuffle=False,
        num_workers=4, pin_memory=True)

#for i, (inputs, targets) in tqdm(enumerate(test_loader0)):
#    print(inputs.size())

test_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AffineMethod,
            normalize,
        ])),
        batch_size=BatchSize, shuffle=False,
        num_workers=4, pin_memory=True)
test_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AffineMethod,
            normalize,
        ])),
        batch_size=BatchSize, shuffle=False,
        num_workers=4, pin_memory=True)
test_loader3 = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AffineMethod,
            normalize,
        ])),
        batch_size=BatchSize, shuffle=False,
        num_workers=4, pin_memory=True)
test_loader4 = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AffineMethod,
            normalize,
        ])),
        batch_size=BatchSize, shuffle=False,
        num_workers=4, pin_memory=True)

#Hash map targets to correct filename
Hash=['a']*9801
for key in AA.class_to_idx:
    Hash[AA.class_to_idx[key]]=key

#print(Hash)
#print(AA.classes)
#print(AA.class_to_idx)


#print(len(test_loader))


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

'''
for i, (inputs, targets) in enumerate(test_loader):
    print(inputs.size())
    #imshow(torchvision.utils.make_grid(inputs))
    #break
    '''



def load_trained_model():
    print('loading trained model ...')
    print('fclayer_affine.t7!')
    model = torch.load('fclayer_affine.t7')
    print(model)
    model.cuda()
    print('done.')
    return model

def predict(model):
    print('predicting ...')


    dic = OrderedDict()
    filename=['a']*9801
    predict=['1']*9801
    #dic['file_name'] = file_name
    #dic['prediction'] = model.predict(X)
    #predicts=[1]*len(file_name)
    tot_loss=0.0
    num=0
    right=0
    #Transpose1=Transpose(type=1)
    #Transpose2=Transpose(type=2)
    outputs=torch.zeros(9801,24).cuda()

    for j in range(0,5):

        if j==0:
            test_loader=test_loader0
        elif j==1:
            test_loader=test_loader1
        elif j==2:
            test_loader=test_loader2
        elif j==3:
            test_loader=test_loader3
        else:
            test_loader=test_loader4

        num=0

        for i, (inputs, targets) in tqdm(enumerate(test_loader)):
            
            inputs=inputs.cuda()
            targets=targets.cuda()
            inputs_var, targets_var = Variable(inputs), Variable(targets)

            #print(inputs.size())
                
            output=model(inputs_var)
            outputs[num:num+inputs.size(0)] = outputs[num:num+inputs.size(0)]+output.data
                
            #indices=indices.view(inputs.size(0))
            for j in range(num,num+inputs.size(0)):
                #filename[j]='test_'+str(targets[j-num])+'.jpg'
                filename[j]=Hash[targets[j-num]]+'.jpg'

            num=num+inputs.size(0)

    _,indices=torch.max(outputs,1)
    indices=indices.view(9801)
    for j in range(0,9801):
            #filename[j]='test_'+str(targets[j-num])+'.jpg'
            
            predict[j]=hash2[indices[j]]

    
    #predicts[9800]=16
    dic['file_name'] = filename
    dic['prediction']=predict
    
    print('done.')
    return pd.DataFrame(dic)


if __name__ == '__main__':
    ## load the test data

    model = load_trained_model()
    # 
    submit = predict(model)
    submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)
