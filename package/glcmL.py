
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

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        #self.conv1 = conv3x3(4,64)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(out.size())
        out = F.relu(self.bn1(self.conv1(x))) # 128
        #print(out.size())
        out = self.layer1(out)#64
        #print(out.size())
        out = self.layer2(out)#32
        #print(out.size())
        out = self.layer3(out)#16
        #print(out.size())
        out = self.layer4(out)#8
        #print(out.size())
        out = F.avg_pool2d(out, 4)#2
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class glcmnet(nn.Module):
    def __init__(self):
        super(glcmnet, self).__init__()
        self.resnet18=torch.load('resnet18_pretrained.t7')
        self.resnet18.fc=nn.Linear(512,100)
        self.resnet18=nn.DataParallel(self.resnet18)
        self.glcmnet=nn.Linear(600,100)
        self.finalLinear=nn.Linear(200,24)


    def forward(self,x1,x2):
        x1=self.resnet18(x1)
        x2=self.glcmnet(x2)
        x=torch.cat((x1,x2),1)#batch*200
        x=self.finalLinear(x)

        return x

def build_glcmnet():
    return glcmnet()

def test():
    model=build_glcmnet().cuda()
    #print(model.vgg11.classifier[6])
    print(model)

    A=Variable(torch.randn(64,3,224,224).cuda())
    B=Variable(torch.randn(64,600).cuda())
    output=model(A,B)
    print(output.data)
    torch.save(model,'../glcmnet_ori.t7')

#test()
#model=torch.load('../glcmnet_ori.t7')
#print(model)
