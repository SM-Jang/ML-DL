import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random


from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    """
    anchor: label 0, 1, 2 -> ranadom one pick -> duplicate(k)
    same label = positive_list -> random sampling(k)
    diff label = negative_list -> random sampling(k)
    """
    def __init__(self, data, label):
        self.data  = data
        self.label = label
        
    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        
        k = 10
        
        anchor_label = random.choice(range(150))
        anchor_label = self.label[anchor_label]
        
        positive_index_set = [  i for i, y in enumerate(self.label) if y == anchor_label]
        negative_index_set = [  i for i, y in enumerate(self.label) if y != anchor_label]
        
        
        # pick anchor in positive index set
        anchor_index = random.choice(positive_index_set)
        positive_index_set.remove(anchor_index)
        
        anchor = self.data[anchor_index] 
        anchor = np.repeat(anchor.reshape(1, -1), k, axis = 0) 
        
        
        # pick positive 
        positive_indices = random.choices(positive_index_set, k = k)
        positive = self.data[positive_indices,:]
        
        # pick negative
        negative_indices = random.choices(negative_index_set, k = k)
        negative = self.data[negative_indices, :]
        
        return anchor, positive, negative, anchor_label
    
    
    
    
    
    


class EmbeddingNet(nn.Module):
    def __init__(self, features, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(features, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim)
        )
        
    def forward(self, x):
        output = self.embedding(x)
        return output


    
    
    
    
    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        
    def forward(self, anchor, positive, negative):
        anchor   = self.embedding_net(anchor)
        positive = self.embedding_net(positive)
        negative = self.embedding_net(negative)
        
        return anchor, positive, negative
    
    def get_embedding(self, x):
        triplet_embedding = self.embedding_net(x)
        return triplet_embedding
    
    def plot_embedding(self, x, loss):
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Triplet Embedding Current Loss: {:4f}'.format(loss))
        ax  = fig.add_subplot(projection='3d')

        x = torch.tensor(x, dtype=torch.float32)

        embeddings = self.get_embedding(x).detach().numpy()
        ax.scatter(
            embeddings[50*0:50*1,0],
            embeddings[50*0:50*1,1],
            embeddings[50*0:50*1,2],
            color='red',
            label='setosa'
        )
        ax.scatter(
            embeddings[50*1:50*2,0],
            embeddings[50*1:50*2,1],
            embeddings[50*1:50*2,2],
            color='blue',
            label='versicolor'
        )
        ax.scatter(
            embeddings[50*2:50*3,0],
            embeddings[50*2:50*3,1],
            embeddings[50*2:50*3,2],
            color='yellow',
            label='virginica'
        )
        ax.legend()
        plt.show()
        plt.close()
        
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(dim=1)
        distance_negative = (anchor - negative).pow(2).sum(dim=1)
        losses = F.relu(distance_positive - 2*distance_negative + self.margin)
        return losses.mean()
