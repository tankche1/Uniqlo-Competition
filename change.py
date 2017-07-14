import os
import pandas as pd
import numpy as np
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


def change():
    data = pd.read_csv('czt.csv', sep=',')
    dict={}
    file_name = []
    output = []
    for row in tqdm(data.iterrows()):
        # print(row)
        f, l = row[1]['file_name'], row[1]['category_id']
        dict[f]=l
    print(dict)

    data2 = pd.read_csv('cky.csv', sep=',')
    for row in tqdm(data2.iterrows()):
        f, l = row[1]['file_name'], row[1]['category_id']
        file_name.append(f)
        if l==0 or l==6 or l==7 or l==18 or l==19 or l==20 or l==21 or l==22:
            output.append(dict[f])
        else:
            output.append(l)
    dic={}
    dic['file_name'] = file_name
    dic['prediction'] = output
    print('done.')
    return pd.DataFrame(dic)

#change()
submit = change()
submit.to_csv('cztcky.csv', index=None, header=None)