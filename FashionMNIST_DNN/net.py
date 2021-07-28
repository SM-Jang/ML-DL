import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    
    def __init__(self, h_in, h1, h2, h3):
        super(AE, self).__init__()

            
        self.encoder =  nn.Sequential(
                nn.Linear(h_in, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, h3)
                )
        self.decoder =  nn.Sequential(
                nn.Linear(h3, h2),
                nn.ReLU(),
                nn.Linear(h2, h1),
                nn.ReLU(),
                nn.Linear(h1, h_in)
                )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded
