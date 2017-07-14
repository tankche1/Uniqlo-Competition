'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning,for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

Transpose=Variable(torch.FloatTensor([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]]
).cuda())


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
    def __init__(self, block, num_blocks, num_classes=24,num_groups=8):
        super(ResNet, self).__init__()
        self.in_planes = 20

        self.conv1 = conv3x3(3,20)
        self.bn1 = nn.BatchNorm2d(20)
        self.layer1 = self._make_layer(block, 20, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 30, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 40, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 60, num_blocks[3], stride=2)
        self.linear = nn.Linear(40*block.expansion*1*1, num_classes)
        self.lineargroup=nn.Linear(30*block.expansion*2*2, num_groups)
        self.finalLinear=nn.Linear(num_groups,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 32*32
        out = self.layer1(out)#16*16
        #print(out.size())
        out = self.layer2(out)#8*8

        group=F.avg_pool2d(out,4)#2*2
        group=group.view(group.size(0),-1)
        group=self.lineargroup(group)

        out = self.layer3(out)#4*4
        #out = self.layer4(out)
        #print(out.size())
        out = F.avg_pool2d(out, 4)
        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(type(Variable(Transpose.cuda())))
        out2=torch.mm(group.cuda(),Transpose)
        out = self.linear(out).cuda()*out2
        return group,out


def ResNet18():
    return ResNet(PreActBlock, [1,1,1])

def test():
    criterion=nn.CrossEntropyLoss()
    net = ResNet18()
    y1,y2 = net(Variable(torch.randn(1,3,32,32)))
    print(y1.size())
    print(y2.size())
    #loss=criterion(y1,Variable(torch.LongTensor(1)))
    #loss.backward()


test()