# heysaik - eeg-eye-state-classification
# https://github.com/heysaik/eeg-eye-state-classification/blob/master/PyTorch_EEGEyeState.ipynb
# Add 5 fold using
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
#from tqdm import tqdm

import warnings
warnings.filterwarnings('always')

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

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]):
    tensors, targets = zip(*data)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets

def print_class(y_train_print, y_test_print):
    print ("----------------------")
    print ("Train data distribution")
    print(f'Train class 0 : {y_train_print.count(0)} about {y_train_print.count(0)/(y_train_print.count(0)+y_train_print.count(1)):.3f}%')
    print(f'Train class 1 : {y_train_print.count(1)} about {y_train_print.count(1)/(y_train_print.count(0)+y_train_print.count(1)):.3f}%')
    print ("----------------------")
    print ("Test data distribution")
    print(f'Test class 0 : {y_test_print.count(0)} about {y_test_print.count(0)/(y_test_print.count(0)+y_test_print.count(1)):.3f}%')
    print(f'Test class 1 : {y_test_print.count(1)} about {y_test_print.count(1)/(y_test_print.count(0)+y_test_print.count(1)):.3f}%')
    print ("----------------------")

def main():

    # Data loading and preprocessing
    data = pd.read_csv('https://raw.githubusercontent.com/plaipmc/Eye_State_FL/main/eeg_eye_state.csv')
    # print(data)
    X = data.iloc[:,0:14]
    y = data.iloc[:,14]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 75 25 
    X_train, X_test, y_train, y_test_last = train_test_split(X, y, test_size=0.25)

    dataset = TensorDataset( torch.FloatTensor(X_train), torch.FloatTensor(y_train.values) )
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    k_folds = 5
    # For fold results
    results = {}

    # Add 5 fold 07/11
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=64, sampler=train_subsampler, collate_fn = collate_fn,drop_last=True) #,collate_fn=lambda x: x
        test_loader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=1, sampler=test_subsampler, collate_fn = collate_fn,drop_last=True)

        y_testp = []
        y_trainp = []
        net = Net()
        net.apply(reset_weights)

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
                optimizer.zero_grad()
                y_trainp.append(y_batch.numpy().tolist())
                y_pred = net(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = accuracy(y_pred, y_batch.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

        print("Finished Training")

        PATH = "./eye_state_net"+ str(fold) +".pth"
        torch.save(net.state_dict(), PATH)

        net = Net()
        net.load_state_dict(torch.load(PATH))
        # (optional) use GPU to speed things up
        net.to(DEVICE)

        y_pred_list = []
        net.eval()
        y_test = []
        correct, total = 0, 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for X_batch,y_batch in test_loader:
                y_test_pred = net(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_testp.append(y_batch.numpy().tolist()[0])

                y_batch_array = y_batch.numpy()
                y_pred_tag_array = y_pred_tag[:].numpy().flatten()
                y_test.append(y_batch_array)
                y_pred_list.append(y_pred_tag_array)

                total += y_batch.size(0)
                correct += (y_pred_tag_array == y_batch_array).sum().item()

        y_trainprint = [int(x[0]) for x in y_trainp]
        y_testprint = [int(x) for x in y_testp]

        
        print('--------------------------------')
        print(f'FOLD {fold}')
        print_class(y_trainprint, y_testprint)
        
        y_pred_list = [j for sub in y_pred_list for j in sub]
        y_test = [i for sub in y_test for i in sub]

        print(classification_report(y_test, y_pred_list))
        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    maxaccfold = 0
    maxacc = 0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
        if maxacc < value :
            maxacc = value
            maxaccfold = key
    print(f'Average: {sum/len(results.items())} %')

    print(f'select model {maxaccfold}')

    # select best model
    PATH = "./eye_state_net"+ str(maxaccfold) +".pth"
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # (optional) use GPU to speed things up
    net.to(DEVICE)

    # Test 
    test_data = testData(torch.FloatTensor(X_test))
    test_loader2 = DataLoader(dataset=test_data, batch_size=1)
    net.eval()
    y_pred_final = []
    with torch.no_grad():
        for X_batch in test_loader2:
            X_batch = X_batch.to(DEVICE)
            y_test_pred = net(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            #print(int(y_pred_tag.cpu()[0,0].tolist()))
            y_pred_final.append(int(y_pred_tag.cpu()[0,0].tolist()))

    #print(len(y_pred_final))
    #print(len(y_test_last))
    #y_trainprint = [int(x[0]) for x in y_trainp]
    #y_testprint = [int(x) for x in y_testp]
    print(classification_report(y_test_last, y_pred_final))
    


if __name__ == "__main__":
    main()
