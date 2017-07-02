# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:45:13 2017

@author: n.aoi
"""

def my_preprocess_method(data, box):
    # write your own preprocess method here
    data_preprocessed = data.crop(box)

    return data_preprocessed