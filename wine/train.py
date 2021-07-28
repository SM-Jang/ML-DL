from data import *
from net import Net
from torch.utils.data import TensorDataset, DataLoader


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument('--check_point', type=str, default='./check_point')
    parser.add_argument('--data_path', type=str, default='./data/wine_data.npy')
    parser.add_argument('--label_path', type=str, default='./data/wine_label.npy')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_arguments()
    
    
    ## Load Data ##
    data = load_data(args.data_path)[:130]
    label = load_data(args.label_path)[:130]
    
    
    ## Split Data ##
    X_train, X_test, y_train, y_test = data_split(data, label, 0.2)

    
    
    ## Data Loader
    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train,
                              batch_size = args.batch_size,
                              shuffle = True)
    test_loader = DataLoader(test,
                            batch_size = args.batch_size,
                            shuffle = False)
    ## Network ##
    h_in = 13
    h1 = 96
    h_out = 2
    
    net = Net(h_in, h1, h_out)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate)
    print(net)


    for epoch in range(args.epochs):
        total_loss = 0

        for X, y in train_loader:

            optimizer.zero_grad()

            output = net(X)

            output = F.sigmoid(output)


            loss = criterion(output, y.long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch+1) % 10 == 0: print("Epoch {}, Loss : {:.4}".format(epoch+1, total_loss))

        

    print("The end!!")
    
    if os.path.isdir(args.check_point) == False: os.mkdir("check_point")
    
    torch.save({
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, args.check_point + '/save.pth')