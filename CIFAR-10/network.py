import torch
import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):
    def __init__(self, h_in, h_out, h1, h2, h3, h4):
        super(network, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels = h_in,
            out_channels = h1,
            kernel_size = 3,
            padding = 1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels = h1,
            out_channels = h2,
            kernel_size = 3,
            padding = 1
        )
        
        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        
        self.fc1 = nn.Linear(16*8*8, h3)
        self.fc2 = nn.Linear(h3, h4)
        self.fc3 = nn.Linear(h4, h_out)
        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        

        x = x.view(-1, 16*8*8)


        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)

        x = F.log_softmax(x, dim = -1)
        
        return x
        
# if __name__ == '__main__':
#     device = torch.device('cuda')
#     x = torch.Tensor(16, 3, 32, 32).to(device)
#     h_in, h_out = 3, 10
#     h1, h2 = 8, 16
#     h3, h4 = 64, 32
#     model = network(h_in, h_out, h1, h2, h3, h4).to(device)
#     print("model", model)
#     y = model(x)
#     print(y[-1])

    