# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:51:07 2017

@author: n.aoi
"""

import os
import pandas as pd
import pickle
import numpy as np
import torch
from PIL import Image
from collections import OrderedDict
from tqdm import *
from termcolor import *
from torch.autograd import Variable
# from package import preprocess_methods # if you used some preprocess method in training phase, you may want to apply it in test phase.

PATH_TO_TEST_IMAGES = os.path.join('data', 'test')
PATH_TO_SUBMIT_FILE = 'submit99.csv'

BatchSize=100

def load_test_data(path_to_test_images):
    print('loading test data ...')
    X = []
    file_name = []
    file = os.listdir(path_to_test_images)
    for f in file:
        try:
            im = Image.open(os.path.join(PATH_TO_TEST_IMAGES, f))
    
            # you may write preprocess method here given an image
            # im = preprocess_methods.my_preprocess_method(im)
            im = im.resize((224,224),Image.ANTIALIAS)
            X.append(np.array(im).transpose(2,0,1))
            file_name.append(f)
        except Exception as e:
            print(str(e))

    X = np.array(X)
    print('done.')
    return X, file_name

def load_trained_model():
    print('loading trained model ...')
    model = torch.load('models/71.8327702703.t7')
    model.cuda()
    print('done.')
    return model

def predict(model, X, file_name):
    print('predicting ...')
    dic = OrderedDict()
    dic['file_name'] = file_name
    #dic['prediction'] = model.predict(X)
    predicts=[1]*len(file_name)
    print(colored(len(file_name),'blue'))
    for i in tqdm(range(0,len(file_name)-1-BatchSize+1+2,BatchSize)):
        #inputs=X[i:i+BatchSize]
        if i+BatchSize<(len(file_name)):
            inputs=(torch.from_numpy(X[i:i+BatchSize])).float()
        else:
            inputs=(torch.from_numpy(X[i:len(file_name))).float()

        inputs=inputs.cuda()
        inputs_var=Variable(inputs)
        output=model(inputs_var)

        _,indices=torch.max(output.data,1)
        indices=indices.view(inputs.size(0))

        for j in range(i,min(i+BatchSize,len(file_name))):
            predicts[j]=indices[j-i]
    
    #predicts[9800]=16

    dic['prediction']=predicts
    print('done.')
    return pd.DataFrame(dic)

if __name__ == '__main__':
    ## load the test data
    X, file_name = load_test_data(PATH_TO_TEST_IMAGES)
    
    ## load the trained model
    model = load_trained_model()
    
    ## output the submit file
    submit = predict(model, X, file_name)
    submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)