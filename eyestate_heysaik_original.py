# heysaik - eeg-eye-state-classification
# https://github.com/heysaik/eeg-eye-state-classification/blob/master/PyTorch_EEGEyeState.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net

# Add library
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader

# (optional) set a fix place so we don't need to download everytime
# DATASET_PATH = "/tmp/nvflare/data/eye_state"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
# DEVICE = "cuda:0"
DEVICE = "cpu"

## Train Data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


## Test Data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def main():

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Data loading and preprocessing
    data = pd.read_csv('https://raw.githubusercontent.com/plaipmc/Eye_State_FL/main/eeg_eye_state.csv')
    print(data)
    X = data.iloc[:,0:14]
    y = data.iloc[:,14]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1556)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(X_test))
    
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    
    net = Net()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # (optional) use GPU to speed things up
    net.to(DEVICE)
    criterion = criterion.to(DEVICE)

    net.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            
            y_pred = net(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = accuracy(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    print("Finished Training")

    PATH = "./cifar_net.pth"
    torch.save(net.state_dict(), PATH)

    net = Net()
    net.load_state_dict(torch.load(PATH))
    # (optional) use GPU to speed things up
    net.to(DEVICE)

    y_pred_list = []
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_test_pred = net(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    print(classification_report(y_test, y_pred_list))


if __name__ == "__main__":
    main()
