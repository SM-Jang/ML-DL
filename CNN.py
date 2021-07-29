import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import  DataLoader, TensorDataset
from keras.datasets.mnist import load_data

batch_size = 32
shuffle    = True
lr         = 0.0005
epochs     = 10

GPU    = 3
plot   = False
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
print("Device:", device)

(x_train, y_train), (x_test, y_test) = load_data()

if plot:
    i = random.randrange(x_train.shape[0])
    sample_data = x_train[i]
    sample_label = y_train[i]
    plt.imshow(sample_data)
    plt.title(sample_label)
    plt.show()
    plt.close()

x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_ds, batch_size=batch_size)


class CNN(nn.Module):
    def __init__(self, h_in, h1, h2, h3, n_classes):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(h_in, h1, kernel_size=3, stride=1),
            nn.ReLU(),
            
            nn.Conv2d(h1, h2, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*24*24, h3),
            nn.ReLU(),
            nn.Linear(h3, n_classes),
        )
    def forward(self, x):
        
        x = self.conv(x)
        x = x.reshape(-1, 16*24*24)
        x = self.classifier(x)
        
        return x
    
h_in, h1, h2, h3, n_classes = 1, 32, 16, 100, 10
model = CNN(h_in, h1, h2, h3, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
print("\n\nTraining Start!!")
for epoch in range(epochs):
    correct = 0
    total   = 0
    
    for x, y in train_loader:
        
        x, y = x.unsqueeze(dim=1).float().to(device), y.long().to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        prediction = torch.argmax(y_pred, axis=1)
        correct += torch.sum(prediction == y).item()
        
        loss.backward()
        optimizer.step()
        
        total += x.shape[0]
    print('Epoch {}/{} Loss {:.4f} Accuracy {:2.2f}'.format(
        epoch+1, epochs, loss.item(), correct/total*100))
    
    
## Model Evaluation ##
model.eval()
correct = 0
total   = 0
for x, y in test_loader:
    x, y = x.unsqueeze(dim=1).float().to(device), y.long().to(device)

    y_pred = model(x)

    prediction = torch.argmax(y_pred, axis=1)

    correct += torch.sum(prediction == y).item()
    total   += x.shape[0]

print('\n\tTest Accuracy is {:2.2f}'.format(correct/total*100))