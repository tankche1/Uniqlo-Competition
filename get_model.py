import torchvision.models as models
import torch

resnet34= models.resnet34(pretrained=True)
print(resnet34)
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')

'''
torch.save(resnet18,'resnet18_pretrained.t7')
'''

'''
alexnet = models.alexnet(pretrained=True)
torch.save(alexnet,'alexnet_pretrained.t7')
print(alexnet)
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')  
print('---------------------------------')
print('---------------------------------')
'''

'''
squeezenet = models.squeezenet1_0(pretrained=True)
print(squeezenet)
torch.save(squeezenet,'squeezenet_pretrained.t7')
'''
'''
vgg16 = models.vgg16(pretrained=True)
print('fuck!!')
#print(vgg16)
torch.save(vgg16,'vgg16_pretrained.t7')
print(vgg16)
'''
