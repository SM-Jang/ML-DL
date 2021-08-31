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

# Linear Variable
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
print('X:', x_train, x_train.shape,
      '\nY:', y_train, y_train.shape)

# plt.title("Linear Data Plot")
# plt.scatter(x_train, y_train, color='black')
# plt.plot(x_train,y_train)
# plt.show()
# plt.close()

# Weight and Bais initialization
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(W), print(b)
optimizer = optim.SGD([W, b], lr=0.01)

print("Learing LR model!")
epochs = 100
for epoch in range(epochs):
    
    # init optim
    optimizer.zero_grad()
    
    # Hypothesis
    h = x_train*W + b
    
    # Cost
    error = (h-y_train) ** 2
    cost = torch.mean( error )

    # optimizing
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch+1, epochs, W.item(), b.item(), cost.item()
        ))