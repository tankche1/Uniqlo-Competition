# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:50:58 2017

@author: n.aoi
"""
# 0,1,2,4,5,7
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
import affine_transforms
# from package.models import MyModel # you may import your own model in the package
# from package import preprocess_methods # you may import your own preprocess method in the package
 
gpu=[0,1,2,4,5,7]

parser=argparse.ArgumentParser(description='uniqlo network')

parser.add_argument('--bh',default='1',
                    help='choose the network')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--batchsize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='SGD weight_decay (default: 5e-4)')
parser.add_argument('--automonous_stopping', type=int,default=0,
                    help='automonous_stopping')
parser.add_argument('--data',default='data',metavar='NT',
                    help='the data directory')
parser.add_argument('--modelpos',default='resnet18_pretrained.t7',metavar='NT',
                    help='the data directory')

parser.add_argument('--load',default=0,type=int,
                    help='choose the network')

args=parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PATH_TO_MODEL = os.path.join('models', 'threshnet',args.bh)

category_num=24
history=[0.01]*1000
historyMax=0.01
Hloss=[0.01]*1000
lossMin=1000000.0
hash2=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9]


# loading pre-trained model

print(colored('initializing the model ...',"blue"))

#model = models.resnet18(pretrained=True)#at least 224*224
from package.threshnet import build_threshnet

if args.load==1:
    model=torch.load('threshnet.t7')
else:
    model=build_threshnet()
print(model)

for param in model.parameters():
    param.requires_grads=False

model.finalLinear.requires_grads=True
model.resnet18.module.fc.requires_grads=True
model.vgg11.module.classifier[6].requires_grads=True
model.Alexnet.module.classifier[6].requires_grads=True
#model.layer4.requires_grads=True
#model.layer3.requires_grads=True
#model.layer2.requires_grads=True

if args.cuda:
    model.cuda()

#print(colored('model ==> ',"green"))
#print(model)

print(colored('initializing done.',"blue"))


# Data loading code

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
traindir='data/train'
valdir='data/val'

train_loader1= torch.utils.data.DataLoader(
                        datasets.ImageFolder(traindir, transforms.Compose([
                            transforms.Scale(224),
                            transforms.ToTensor(),
                            affine_transforms.Affine(rotation_range=10,translation_range=[0.1,0.1],shear_range=0.1,zoom_range=[0.9,1.1]),
                            normalize,
                        ])),
                        batch_size=args.batchsize, shuffle=True,
                        num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchsize, shuffle=False,
        num_workers=4, pin_memory=True)

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)

def save_model(model, name):
    print('saving the model ...')
    if not os.path.exists(PATH_TO_MODEL):
        os.mkdir(PATH_TO_MODEL)

    torch.save(model,PATH_TO_MODEL+'/'+str(historyMax)+'.t7')
    print('done.')

def visualize(data):
    for i in range(0,24):
        print('color '+str(i)+ '!')
        for j in range(0,24):
            if(data[i][j]>5):
                print '[%d][%d] = %.2f ' %(i,j,data[i][j]) ,
        print()
    for i in range(0,24):
        print(colored('color %d precision: %.2f !' %(i,data[i][i]),'green'))

def train(epoch):
    train_loader= torch.utils.data.DataLoader(
                        datasets.ImageFolder(traindir, transforms.Compose([
                            transforms.Scale(224),
                            transforms.ToTensor(),
                            affine_transforms.Affine(rotation_range=10,translation_range=[0.1,0.1],shear_range=0.1,zoom_range=[0.9,1.1]),
                            normalize,
                        ])),
                        batch_size=args.batchsize, shuffle=True,
                        num_workers=4, pin_memory=True)
    model.train()
    print(colored('training epoch '+ str(epoch) + ' !','blue'))

    print(colored('loading data!','green'))
    
    print(colored('done!','green'))
    tot_loss=0.0
    num=0
    right=0

    for i, (inputs, targets) in tqdm(enumerate(train_loader)):
        #print(inputs.size())
        if args.cuda:
            inputs=inputs.cuda(async=True)
            targets=targets.cuda(async=True)  

        inputs_var, targets_var = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = model(inputs_var)
        
        loss=criterion(outputs, targets_var)/(args.batchsize*1.0)

        loss.backward()
        optimizer.step()

        tot_loss=tot_loss+loss.data[0]
        num=num+inputs.size(0)

        _,indices=torch.max(outputs.data,1)
        indices=indices.view(inputs.size(0))
        right=right+sum(indices==targets)
        
    #print(output.data)
    #averageloss=2.3  
    #averageloss=(tot_loss*1.0)/(num*1.00)
    print(colored("totloss: %.8f ! " %tot_loss,'red'))

    precision=2.3
    precision=(right*100.0)/(num*1.00)
    print(colored("precision: %.2f%c ! " %(precision,'%'),'red'))

    global Hloss,lossMin
    Hloss[epoch]=tot_loss
    if epoch==1:
        lossMin=tot_loss
    else:
        lossMin=min(lossMin,tot_loss)

    #print(colored("right: %d  ! " %right,'red'))

def test(epoch):
    Miss=[[0 for col in range(0,24)] for row in range(0,24)]# suppose i to be j
    model.eval()
    print(colored('Testing!','blue'))

    tot_loss=0.0
    num=0
    right=0

    for i, (inputs, targets) in tqdm(enumerate(val_loader)):

        if args.cuda:
            inputs=inputs.cuda(async=True)
            targets=targets.cuda(async=True)  

        inputs_var, targets_var = Variable(inputs), Variable(targets)

        #optimizer.zero_grad()
        outputs = model(inputs_var)
        
        loss=criterion(outputs, targets_var)/(args.batchsize*1.0)

        #loss.backward()
        #optimizer.step()

        tot_loss=tot_loss+loss.data[0]
        num=num+inputs.size(0)

        _,indices=torch.max(outputs.data,1)
        indices=indices.view(inputs.size(0))
        right=right+sum(indices==targets)

        for j in range(0,inputs.size(0)):
            Miss[hash2[int(targets[j])]][hash2[int(indices[j])]]+=1
        
    #print(output.data)
    #averageloss=2.3  
    #averageloss=(tot_loss*1.0)/(num*1.00)
    print(colored("totloss: %.8f ! " %tot_loss,'red'))

    precision=2.3
    precision=(right*100.0)/(num*1.00)
    print(colored("precision: %.2f%c ! " %(precision,'%'),'red'))

    global historyMax,history
    history[epoch]=precision
    historyMax=max(historyMax,precision)

    Misspre=np.array([[0.01 for col in range(0,24)] for row in range(0,24)])
    for i in range(0,24):
        for j in range(0,24):
            Misspre[hash2[i]][hash2[j]]=(100*Miss[hash2[i]][hash2[j]]*1.000)/(sum(Miss[hash2[i]])*1.000)
    #torch.save(Misspre,'visualize/visualize.t7')
    #visualize(Misspre)



if __name__ == '__main__':
    

    #save_model(model, PATH_TO_MODEL)

    #print(np.shape(X))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if historyMax==history[epoch]:
            save_model(model, PATH_TO_MODEL)
        if args.automonous_stopping==1:
            haha=0
            if(epoch>10):
                for i in range(epoch-10+1,epoch+1):
                    if(Hloss[i]==lossMin):
                        haha=1
                if haha==0:
                    break

