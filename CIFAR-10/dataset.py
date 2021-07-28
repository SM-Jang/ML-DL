import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data Download and Build Data Loader

def get_cifar10(path):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train = datasets.CIFAR10(root=path,
                            train = True,
                            download=True,
                            transform=transform)
    
    test = datasets.CIFAR10(root=path,
                            train = False,
                            download=False,
                            transform=transform)
    
    return train, test

def get_loader(train, test, bs):
    train_loader = DataLoader(dataset=train,
                             batch_size=bs,
                             shuffle=True)
    
    test_loader = DataLoader(dataset=test,
                             batch_size=bs,
                             shuffle=False)
    
    return train_loader, test_loader

if __name__  == '__main__':
    current = os.getcwd()
    if os.path.isdir(current + '/data') == False: os.mkdir('data')
        
    get_cifar10(current + '/data')
        