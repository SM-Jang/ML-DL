import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split



## Set Parameters ##
split      = 0.2
batch_size = 16
lr         = 0.005
epochs     = 50
n_classes  = 10
plot       = False


## DATA EDA ##
digits = load_digits()
print(digits.DESCR)

## Load Data ##
data = digits.data
labels = digits.target


## Visualization ##
if plot:
    images = digits.images
    print(images.shape)

    rows = 3
    cols = 3
    fig, axes = plt.subplots(3,3)
    fig.set_size_inches(10,10)
    for r in range(rows):
        for c in range(cols):
            axes[r][c].imshow(images[3*r+c])
            axes[r][c].set_title("Label:{}".format(labels[3*r+c]))

    plt.show()
    plt.close()

## Torch Dataset ##
x, y = torch.from_numpy(data), torch.from_numpy(labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, shuffle=True )

print("Train Dataset Size:", x_train.shape, y_train.shape)
print("Test Dataset Size:", x_test.shape, y_test.shape)

train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)


train_loader = DataLoader(train_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

## MLP Model ##
h_in, h1, h2 = x.shape[1], 32, 16


model = nn.Sequential(
    nn.Linear(h_in, h1),
    nn.ReLU(),
    
    nn.Linear(h1, h2),
    nn.ReLU(),
    
    nn.Linear(h2, n_classes)
)


## Optimizer ##
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


## Model Train ##
model.train()
print("\n\nTraining Start!!")
for epoch in range(epochs):
    correct = 0
    total   = 0
    
    for x, y in train_loader:
        x, y = x.float(), y.long()
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        
        prediction = torch.argmax(pred, axis=1)
        
        correct += torch.sum(prediction == y).item()
        total   += x.shape[0]
        
        loss.backward()
        optimizer.step()
        

    print('Epoch {}/{} Loss {:.4f} Accuracy {:2.2f}'.format(
        epoch+1, epochs, loss.item(), correct/total*100))
    
    
## Model Evaluation ##
model.eval()
correct = 0
total   = 0
for x, y in test_loader:
    x, y = x.float(), y.long()

    pred = model(x)

    prediction = torch.argmax(pred, axis=1)

    correct += torch.sum(prediction == y).item()
    total   += x.shape[0]

print('Test Accuracy is {:2.2f}'.format(correct/total*100))