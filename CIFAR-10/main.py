import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from network import network
from dataset import get_cifar10, get_loader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--check_point', type=str, default='./check_point')
    parser.add_argument('--data_path', type=str, default='./data')
    
    parser.add_argument('--mode', type=str)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    
    return parser.parse_args()


def train_model(model, train_loader, device, optimizer, criterion, log_interval):
    model.train()
    for batch_idx, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('[{} / {}] ({:.0f}%)]\tTrain Loss {:.4f}'.format(
            batch_idx * len(img), len(train_loader.dataset),
            batch_idx / len(train_loader) * 100., loss.item()))
            

def test_model(model, path, device, test_loader):
    
    ## load the save model ##
    checkpoint = torch.load(path+'/Cifar10_classifier.pth')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    total = 0
    correct = 0 
    
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, predicted = torch.max(output, 1)
            
            total += label.size(0)
            correct += (predicted ==label).sum().item()
        
    print('The Final Test Accuracy is {:2.2f}% \t'.format((100 * correct / total)))

    



if __name__ == '__main__':
    
    ## parser ##
    args = get_arguments()
    
    ## data dir ##
    if os.path.isdir(args.data_path) == False:
        print("Make Data Directory")
        os.mkdir('data') 
    ## check_point dir ##
    if os.path.isdir(args.check_point) == False: 
        print("Make Check Point Directory")
        os.mkdir("check_point")
        

    ## device ##
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    print("The device is", device)    
    
    ## dataset and loader ##
    train, test = get_cifar10(args.data_path)
    train_loader, test_loader = get_loader(train, test, args.batch_size)
    img_x, img_y, channel, num_classes = 32, 32, 3, 10
    print(train, test)
    print("Image Shape is (%s, %s, %s)" %(channel, img_x, img_y))
    
    ## network ##
    h_in, h_out = channel, num_classes
    h1, h2 = 8, 16
    h3, h4 = 64, 32
    model = network(h_in, h_out, h1, h2, h3, h4).to(device)
    print(model)
    
    ## Loss & Optimizer ##
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr)
    print("Loss is", criterion)
    print("Optimizer is", optimizer)
    
    ## Train or Test the model ##
    log_interval = 100
    
    if args.mode == 'train':
        print("Train Mode")
        for epoch in range(args.epochs):
            print("Total Epochs: [%s / %s]" %(epoch+1, args.epochs))
            train_model(model, train_loader, device, optimizer, criterion, log_interval)
            
            
        ## save the trained model
        print("The training is ended!!")
        torch.save({
            'model':model.state_dict()
        }, args.check_point + '/Cifar10_classifier.pth')
        
        
    if args.mode == 'test':
        print('Test Model')
        checkpoint = torch.load('./check_point/Cifar10_classifier.pth')
        model.load_state_dict(checkpoint['model'])
        
        test_model(model, args.check_point, device, test_loader)
    