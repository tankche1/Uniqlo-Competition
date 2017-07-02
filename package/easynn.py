import torch
import torch.nn as nn
import torch.nn.functional as F

w_bound=0.5

def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        #print(m.weight.data)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        #print(m.weight.data)
    

class easynn(nn.Module):
	def __init__(self,category_num):
		super(easynn, self).__init__()
		self.conv1=nn.Conv2d(3,10,3,stride=1,padding=1)
		self.conv2=nn.Conv2d(10,20,3,stride=1,padding=1)
		self.conv3=nn.Conv2d(20,20,3,stride=1,padding=1)

		self.fc1=nn.Linear(32*32*20,32*32)
		self.fc2=nn.Linear(32*32,category_num)

		self.apply(weights_init)

	def forward(self,x):
		x=F.relu(F.max_pool2d(self.conv1(x), 2)) # 64*64
		x=F.relu(F.max_pool2d(self.conv2(x), 2)) # 32*32
		x=F.relu(self.conv3(x)) # 32*32

		x=x.view(-1,32*32*20)

		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=F.log_softmax(x)
		return x

        