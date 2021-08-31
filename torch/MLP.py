# Package Setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

seed = 1
torch.manual_seed(seed)

print('Current Pytorch Version:',torch.__version__)


# Multiple X Variables -> 4samples, 2dim
x1 = torch.randn(1,2)
x2 = torch.randn(1,2)
x3 = torch.randn(1,2)
x4 = torch.randn(1,2)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)

# Target y Variable
y = torch.randn(4,2)
print(y.shape)

# Make one matrix data
# X = torch.stack([x1,x2,x3,x4]).squeeze()

# plt.scatter(X[:,0], y[:,0], label='Dim1')
# plt.scatter(X[:,1], y[:,1], label='Dim2')
# plt.legend()
# plt.show()
# plt.close()

# Weight and Bais initialization
W1 = torch.zeros(1, requires_grad=True)
W2 = torch.zeros(1, requires_grad=True)
W3 = torch.zeros(1, requires_grad=True)
W4 = torch.zeros(1, requires_grad=True)

b = torch.zeros(1, requires_grad=True)


optimizer = optim.SGD([W1, W2, W3, W4, b], lr=0.01)

print("Learing MLP model!")
epochs = 100
for epoch in range(epochs):
    
    # init optim
    optimizer.zero_grad()
    
    # Hypothesis
    h = x1*W1 + x2*W2 + x3*W3 + x4*W4 + b
    
    # Cost
    error = (h-y) ** 2
    cost = torch.mean( error )

    # optimizing
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, W3: {:.3f}, W4: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, epochs, W1.item(), W2.item(), W3.item(), W4.item(), b.item(), cost.item()
        ))