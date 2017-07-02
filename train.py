# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:50:58 2017

@author: n.aoi
"""

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
# from package.models import MyModel # you may import your own model in the package
# from package import preprocess_methods # you may import your own preprocess method in the package


parser=argparse.ArgumentParser(description='uniqlo network')

parser.add_argument('--nettype',default='easynn2',metavar='NT',
                    help='choose the network')
parser.add_argument('--bh',default='1',
                    help='choose the network')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

args=parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PATH_TO_TRAIN_IMAGES = os.path.join('data', 'processed32', 'processed_train_images')
PATH_TO_TRAIN_DATA = os.path.join('data', 'given', 'train_master.tsv')
PATH_TO_MODEL = os.path.join('models', args.nettype)

category_num=24

print(colored('initializing the model ...',"blue"))

if args.nettype=='easynn':
    from package.easynn import easynn
    model=easynn(category_num)
elif args.nettype=='easynn2':
    from package.easynn2 import easynn2
    model=easynn2(category_num)

if args.cuda:
    model.cuda()

print(colored('model ==> ',"green"))
print(model)

print(colored('initializing done.',"blue"))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def load_train_data(path_to_train_images, path_to_train_data):
    print('loading train data ...')
    data = pd.read_csv(path_to_train_data, sep='\t')
    X = []
    y = []
    for row in tqdm(data.iterrows()):
        f, l = row[1]['file_name'], row[1]['category_id']
        try:
            im = Image.open(os.path.join(path_to_train_images, f))

            # you may write preprocess method here given an image
            # im = preprocess_methods.my_preprocess_method(im)

            X.append(np.array(im).transpose(2,0,1))
            y.append(l)
        except Exception as e:
            print(str(e))

    X = np.array(X)
    y = np.array(y)
    print('done.')
    return X, y

def save_model(model, name):
    print('saving the model ...')

    torch.save(model,PATH_TO_MODEL+args.bh+'.t7')
    print('done.')



def train(epoch,X,y):
    model.train()
    print(colored('training epoch '+ str(epoch) + ' !','blue'))

    tot_loss=0.0
    num=0
    right=0
    
    for i in tqdm(range(1,X.shape[0]-args.batchsize+2,args.batchsize)):
        inputs=(torch.from_numpy(X[i:i+args.batchsize])).float()
        targets=(torch.from_numpy(y[i:i+args.batchsize]))

        #print(type(inputs))
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        #print(inputs.size())
        optimizer.zero_grad()
        output = model(inputs)
        
        #print(output.data)
        #print(indices)
        #print(targets)
        #print(sum(indices==targets))
        #print(output.size())
        loss =F.nll_loss(output, targets)

        tot_loss=tot_loss+sum(loss.data)
        #print(loss.data)
        Smax,indices=torch.max(output.data,1)
        indices=indices.view(args.batchsize)
        right=right+sum(indices==targets.data)

        loss.backward()
        optimizer.step()
        num=num+1
        

        #print(output.data.size())
        #print(targets.data.size())

        
        #print(type(indices))
        
        #print(indices,targets)
        
        #print(sum(indices==targets.data))
        #print(type(right))
    
    #print(indices,targets.data)
    print(output.data)
    averageloss=2.3  
    averageloss=(tot_loss*1.0)/(num*args.batchsize*1.00)
    print(colored("averageloss: %.4f ! " %averageloss,'red'))
    precision=2.3
    precision=(right*100.0)/(num*args.batchsize*1.00)

    print(colored("precision: %.2f%c ! " %(precision,'%'),'red'))

    print(colored("right: %d  ! " %right,'red'))





if __name__ == '__main__':
    ## load the data for training
    X, y = load_train_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA)
    
    ## instanciate and train the model
    #model = get_model()

    ## save the trained model
    save_model(model, PATH_TO_MODEL)

    #print(np.shape(X))

    for epoch in range(1, args.epochs + 1):
        train(epoch,X,y)
        if epoch%5==0:
            save_model(model, PATH_TO_MODEL)