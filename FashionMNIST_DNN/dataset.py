#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

    
    
def get_dataset(path):
    train = datasets.FashionMNIST(root=path,
                            train = True,
                            download=True,
                            transform = transforms.ToTensor())
    test = datasets.FashionMNIST(root=path,
                            train = False,
                            download=False,
                            transform = transforms.ToTensor())
    
    return train, test
    
    
def get_loader(train, test, bs):
    train_loader = DataLoader(dataset=train,
                             batch_size=bs,
                             shuffle=True)
    
    test_loader = DataLoader(dataset=test,
                             batch_size=bs,
                             shuffle=False)
    
    return train_loader, test_loader
    
 
