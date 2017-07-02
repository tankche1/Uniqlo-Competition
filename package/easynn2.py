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


class easynn2(nn.Module):
	def __init__(self,category_num):
		super(easynn2,self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(500, 100)
		self.fc2 = nn.Linear(100, category_num)

		self.apply(weights_init)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))#14
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))# 5
		x = x.view(-1, 500)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x)

