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

# (1) import nvflare client API
import nvflare.client as flare

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


def accuracy_func(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accu = correct_results_sum/y_test.shape[0]
    accu = torch.round(accu * 100)
    
    return accu


def main():
    global accuracy

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
    
    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (4) loads model from NVFlare
        net.load_state_dict(input_model.params)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

        # (optional) use GPU to speed things up
        net.to(DEVICE)
        criterion = criterion.to(DEVICE)

        steps = EPOCHS * len(train_loader)

        for e in range(1, EPOCHS+1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                
                y_pred = net(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = accuracy_func(y_pred, y_batch.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(net.state_dict(), PATH)

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            net = Net()
            net.load_state_dict(input_weights)
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

        # (6) evaluate on received model for model selection
        accuracy = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
