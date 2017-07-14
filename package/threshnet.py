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



class threshnet(nn.Module):
    def __init__(self):
        super(threshnet, self).__init__()
        self.resnet18=nn.DataParallel(torch.load('resnet18_pretrained.t7'))
        self.Alexnet=nn.DataParallel(torch.load('alexnet_pretrained.t7'))
        self.vgg11=nn.DataParallel(torch.load('vgg11_pretrained.t7'))
        self.finalLinear=nn.Linear(3000,24)
        #self.resnet18.__init__()
        #self.Alexnet.__init__()
        #self.vgg11.__init__()

    def forward(self,x):
        x1=self.resnet18(x)
        x2=self.Alexnet(x)
        x3=self.vgg11(x)
        out=torch.cat((x1,x2,x3),1)#batch*3000
        out=self.finalLinear(out)
        return out




def build_threshnet():
    #resnet18=torch.load('../resnet18_pretrained.t7')
    #Alexnet=torch.load('../alexnet_pretrained.t7')
    #vgg11=torch.load('../vgg11_pretrained.t7')

    #resnet18.fc=nn.Linear(512,24)
    #Alexnet=nn.Sequential(Alexnet,nn.Linear(1000,24))
    #vgg11=nn.Sequential(vgg,nn.Linear(1000,24))

    return threshnet()
'''
model=build_threshnet().cuda()
#print(model.vgg11.classifier[6])
print(model)

A=Variable(torch.randn(64,3,224,224).cuda())
output=model(A)
print(output.data)
'''

