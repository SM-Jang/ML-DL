import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, h_in, h1, h_out):
        super(Net, self).__init__()
        
        self.h_in, self.h_out = h_in, h_out
        self.h1 = h1
        
        self.fc1 = nn.Linear(h_in, h1)
        self.fc2 = nn.Linear(h1, h_out)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
        