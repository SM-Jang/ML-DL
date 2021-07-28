from net import AE
from dataset import get_dataset, get_loader

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

label_tags = {
    0: 'T-Shirt', 
    1: 'Trouser', 
    2: 'Pullover', 
    3: 'Dress', 
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt',
    7: 'Sneaker', 
    8: 'Bag', 
    9: 'Ankle Boot'
}

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--check_point', type=str, default ='./check_point')
    parser.add_argument('--data_path', type=str, default='./data')
    
    parser.add_argument('--mode', type=str)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    
    return parser.parse_args()


def train_model(model, train_loader, img_x, img_y,
          criterion, optimizer, device, log_interval):
    
    model.train()
    for batch_idx, (img, y) in enumerate(train_loader):
        img = img.view(-1, img_x * img_y).to(device)
        target = img.view(-1, img_x * img_y).to(device)
        
        optimizer.zero_grad()
        enc, dec = model(img)
        
        loss = criterion(dec, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print("Train Epoch {}\t Train Loss {:.4f}".format(batch_idx, loss.item()))
            

    return model
    
  

    
def test_model(model, path, test_loader, img_x, img_y, device, criterion):
    ## load the save model ##
    checkpoint = torch.load(path+'/Autoencoder.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    test_loss = 0
    real_img, gen_img, label = list(), list(), list()
    
    with torch.no_grad():
        for img, y in test_loader:
            img = img.view(-1, img_x * img_y).to(device)
            target = img.view(-1, img_x * img_y).to(device)
            
            enc, dec = model(img)
            
            test_loss += criterion(dec, target).item()
            
            real_img.append(img.to('cpu'))
            gen_img.append(dec.to('cpu'))
            label.append(y)
            
    test_loss /= len(test_loader.dataset)
    
    return test_loss, real_img, gen_img, label

if __name__ == '__main__':
    
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

    
    ## dataset ##
    train, test = get_dataset(args.data_path)
    train_loader, test_loader = get_loader(train, test, args.batch_size)
    img_x, img_y = 28, 28
    print(train, test)
    print("Img Size is (%s, %s)" %(img_x, img_y))
    
    ## Network ##
    h_in = img_x * img_y
    h1, h2, h3 = 512, 256, 128
    model = AE(h_in, h1, h2, h3).to(device)
    print(model)
    
    ## loss, Optimizer ##
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    ## Train or Test model ##
    log_interval = 100
    
    if args.mode == 'train':
        print('Train Mode')
        for i in range(args.epochs):
            print('Total Epochs: [%s / %s]' %(i+1, args.epochs))
            train_model(model, train_loader, img_x, img_y, criterion, optimizer, device, log_interval)
        
        # Save the trained model
                    
        print("Training End!!")
        torch.save({
        'model': model.state_dict(),
         }, args.check_point + '/Autoencoder.pth')  
            
            
        
    if args.mode == 'test':
        print('Test Mode')


        test_loss, real_img, gen_img, label = test_model(model, args.check_point, test_loader, img_x, img_y, device, criterion)
        
        print("The test loss is {:.4f}".format(test_loss))
        
        rows, cols = 2, 10
        
        fig, axes = plt.subplots(rows, cols, figsize=(10,4))
        
        for i in range(10):
            img = np.reshape(real_img[0][i], (28, 28))
            axes[0][i].imshow(img, cmap='gray_r')
            axes[0][i].set_xticks(())
            axes[0][i].set_yticks(())
            axes[0][i].set_title(label_tags[label[0][i].item()])
            
        for i in range(10):
            img = np.reshape(gen_img[0][i], (28, 28))
            axes[1][i].imshow(img, cmap='gray_r')
            axes[1][i].set_xticks(())
            axes[1][i].set_yticks(())
            axes[1][i].set_title(label_tags[label[0][i].item()])
            
        plt.show()
        fig.savefig('./FashionMNIST.png')
        
        
        
        

    