import torchvision.models as models
import torch
'''
resnet34= models.resnet34(pretrained=True)
print(resnet34)
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')
print('---------------------------------')


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



squeezenet = models.squeezenet1_0(pretrained=True)
print(squeezenet)
torch.save(squeezenet,'squeezenet_pretrained.t7')
'''

vgg11 = models.vgg11(pretrained=True)
print(vgg11)
torch.save(vgg11,'vgg11_pretrained.t7')

