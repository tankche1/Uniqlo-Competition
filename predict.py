# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:51:07 2017

@author: n.aoi
"""

import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from collections import OrderedDict
# from package import preprocess_methods # if you used some preprocess method in training phase, you may want to apply it in test phase.

PATH_TO_TRAINED_MODEL = os.path.join('models', 'mlp.pkl')
PATH_TO_TEST_IMAGES = os.path.join('data', 'processed', 'test_images')
PATH_TO_SUBMIT_FILE = 'submit.csv'

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
            
            X.append(np.array(im).flatten())
            file_name.append(f)
        except Exception as e:
            print(str(e))

    X = np.array(X)
    print('done.')
    return X, file_name

def load_trained_model(path_to_trained_model):
    print('loading trained model ...')
    with open(path_to_trained_model, mode='rb') as f:
        model = pickle.load(f)
    print('done.')
    return model

def predict(model, X, file_name):
    print('predicting ...')
    dic = OrderedDict()
    dic['file_name'] = file_name
    dic['prediction'] = model.predict(X)
    print('done.')
    return pd.DataFrame(dic)

if __name__ == '__main__':
    ## load the test data
    X, file_name = load_test_data(PATH_TO_TEST_IMAGES)
    
    ## load the trained model
    model = load_trained_model(PATH_TO_TRAINED_MODEL)
    
    ## output the submit file
    submit = predict(model, X, file_name)
    submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)