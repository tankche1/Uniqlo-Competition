# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:49:26 2017

@author: n.aoi
"""
# 667*667*3

import os
import zipfile
from PIL import Image
from package.preprocess_methods import my_preprocess_method
from scipy.misc import imread,imsave
from numpy import *
from tqdm import *

PATH_TO_GIVEN_DATA = os.path.join('data','given')
PATH_TO_MY_DATA = os.path.join('data','processed32')
if not os.path.exists(PATH_TO_MY_DATA):
    os.mkdir(PATH_TO_MY_DATA)

PATH_TO_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, 'train')
PATH_TO_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'test')

PATH_TO_DST_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_train_images')
PATH_TO_DST_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_test_images')
if not os.path.exists(PATH_TO_DST_TRAIN_IMAGES):
    os.mkdir(PATH_TO_DST_TRAIN_IMAGES)
if not os.path.exists(PATH_TO_DST_TEST_IMAGES):
    os.mkdir(PATH_TO_DST_TEST_IMAGES)

def extract_zipfile(src, dst):
    print('extracting '+src+' to '+dst+' ...')
    with zipfile.ZipFile(src, 'r') as zip_file:
        zip_file.extractall(dst)
    print('done.')

def preprocess_image(path_to_images, path_to_dst_images):
    print('preprocessing  '+ path_to_images + ' to ' + path_to_dst_images)
    files = os.listdir(path_to_images)
    for f in files:
        im = Image.open(os.path.join(path_to_images, f))
        
        # you may write your own preprocess method here

        #crop_size = (4,4,24,24)
        im = im.resize((32,32),Image.ANTIALIAS)
        im.save(os.path.join(path_to_dst_images, f))
    
    # if you want to process batch-wise, write the process here

def lookup(path_to_images):
    print('start lookup at'+ path_to_images )
    files=os.listdir(path_to_images)
    for f in tqdm(files):
        im=imread(os.path.join(path_to_images, f))
        #print(f + ' : ',shape(im))
        #print(shape(im))
        #print(shape(im))

    print('end lookup!  ')


if __name__ == '__main__':
    ## extract the given zipfiles(or maybe tarfiles)

    if not os.path.exists(PATH_TO_TRAIN_IMAGES):
        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, 'train.zip'), PATH_TO_MY_DATA)
    if not os.path.exists(PATH_TO_TEST_IMAGES):
        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, 'test.zip'), PATH_TO_MY_DATA)
    
    ##preprocess the data (maybe optional)
    #if not os.path.exists(PATH_TO_DST_TRAIN_IMAGES):
    preprocess_image(PATH_TO_TRAIN_IMAGES, PATH_TO_DST_TRAIN_IMAGES)
    #if not os.path.exists(PATH_TO_DST_TEST_IMAGES):
    preprocess_image(PATH_TO_TEST_IMAGES, PATH_TO_DST_TEST_IMAGES)

    #lookup(PATH_TO_DST_TEST_IMAGES)