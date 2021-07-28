import numpy as np
import torch
import os

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def save_wine(path):
    wine = load_wine()

    np.save( path + '/wine_data.npy', wine.data)
    np.save( path + '/wine_label.npy', wine.target )

def load_data(path):
    return np.load(path)

def data_split(data, label, size):
    """
    nd.array를 train, test로 나누어서
    torchdml tensor로 변환
    """
    index = len(data)
    index = np.arange(index)
    train, test = train_test_split(index, test_size = size)
    train_data, train_label = torch.from_numpy(data[train]).float(), torch.from_numpy(label[train]).long()
    test_data, test_label = torch.from_numpy(data[test]).float(), torch.from_numpy(label[test]).long()
    
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    
    path = './data'
    
    if os.path.isdir(path) == False: os.mkdir('data') 
    
    save_wine(path)