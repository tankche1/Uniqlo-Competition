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
import torch.nn as nn
# from package.models import MyModel # you may import your own model in the package
# from package import preprocess_methods # you may import your own preprocess method in the package


parser=argparse.ArgumentParser(description='uniqlo network')

parser.add_argument('--nettype',default='resnet18',metavar='NT',
                    help='choose the network easynn|easynn2|resnet18')
parser.add_argument('--bh',default='1',
                    help='choose the network')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='SGD weight_decay (default: 5e-4)')

args=parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PATH_TO_TRAIN_IMAGES = os.path.join('data', 'processed32', 'train_images')
PATH_TO_TRAIN_DATA = os.path.join('data', 'given', 'train_master.tsv')

PATH_TO_TEST_IMAGES = os.path.join('data', 'processed32', 'test_images')
PATH_TO_TEST_DATA = os.path.join('data', 'given', 'test_master.tsv')
PATH_TO_MODEL = os.path.join('models', args.nettype)

category_num=24

print(colored('initializing the model ...',"blue"))

if args.nettype=='easynn':
    from package.easynn import easynn
    model=easynn(category_num)
elif args.nettype=='easynn2':
    from package.easynn2 import easynn2
    model=easynn2(category_num)
elif args.nettype=='resnet18':
    from package.resnet import ResNet18
    model=ResNet18()

if args.cuda:
    model.cuda()

print(colored('model ==> ',"green"))
print(model)

print(colored('initializing done.',"blue"))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)

if args.nettype=='easynn':
    criterion=F.nll_loss()
elif args.nettype=='easynn2':
    criterion=F.nll_loss()
elif args.nettype=='resnet18':
    criterion = nn.CrossEntropyLoss()

def load_data(path_to_train_images, path_to_train_data,op):
    if op=='train':
        print('loading train data ...')
    elif op=='test':
        print('loading test data ...')
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

        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        output = model(inputs)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        tot_loss=tot_loss+loss.data[0]
        _,indices=torch.max(output.data,1)
        indices=indices.view(args.batchsize)
        right=right+sum(indices==targets.data)
        num=num+1
        
    #print(output.data)
    averageloss=2.3  
    averageloss=(tot_loss*1.0)/(num*args.batchsize*1.00)
    print(colored("averageloss: %.4f ! " %averageloss,'red'))
    precision=2.3
    precision=(right*100.0)/(num*args.batchsize*1.00)

    print(colored("precision: %.2f%c ! " %(precision,'%'),'red'))

    print(colored("right: %d  ! " %right,'red'))

def test(epoch,X,y):
    model.eval()
    print(colored('Testing!','blue'))

    tot_loss=0.0
    num=0
    right=0
    
    for i in tqdm(range(1,X.shape[0]-args.batchsize+2,args.batchsize)):
        inputs=(torch.from_numpy(X[i:i+args.batchsize])).float()
        targets=(torch.from_numpy(y[i:i+args.batchsize]))

        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        
        output = model(inputs)
        
        loss = criterion(output, targets)

        tot_loss=tot_loss+loss.data[0]
        _,indices=torch.max(output.data,1)
        indices=indices.view(args.batchsize)
        right=right+sum(indices==targets.data)
        num=num+1
        
    #print(output.data)
    averageloss=2.3  
    averageloss=(tot_loss*1.0)/(num*args.batchsize*1.00)
    print(colored("averageloss: %.4f ! " %averageloss,'red'))
    precision=2.3
    precision=(right*100.0)/(num*args.batchsize*1.00)

    print(colored("precision: %.2f%c ! " %(precision,'%'),'red'))

    print(colored("right: %d  ! " %right,'red'))



if __name__ == '__main__':
    ## load the data for training
    X, y = load_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA,'train')

    testX, testy=load_data(PATH_TO_TEST_IMAGES, PATH_TO_TEST_DATA,'test')
    
    ## instanciate and train the model
    #model = get_model()

    ## save the trained model
    save_model(model, PATH_TO_MODEL)

    #print(np.shape(X))

    for epoch in range(1, args.epochs + 1):
        train(epoch,X,y)
        test(epoch,testX,testy)
        if epoch%5==0:
            save_model(model, PATH_TO_MODEL)
