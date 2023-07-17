# use the raw data :')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# some hyperparameter type things :)
sr = 200
t = 200
ch = 19
affix = '_raw'

# ConvNet_Finger
class ConvNet_Finger(nn.Module):
    def __init__(self):
        super(ConvNet_Finger, self).__init__()
        ### FILL IN ### [10 POINTS]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (7, 3)),
            nn.ReLU(), 
            #nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (5, 3)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )

        self.hidden = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(121728, 256),
            nn.Linear(43456, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # all features norefer
            # nn.Linear(1408, 256), #pseudosampled featrues .
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        ### FILL IN ### [5 POINTS]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.hidden(x)
        return x

# final NeuralNet
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        ### FILL IN ### [10 POINTS]

        self.hidden = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(121728, 256),
            nn.Linear(43456, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # all features norefer
            # nn.Linear(1408, 256), #pseudosampled featrues .
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        nn.Sequential(
            nn.Linear(5, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 5)
        )
        return x

# helper functions

def train(train_loader, n_epochs):
    for epoch in range(n_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # print('batch = ', batch_idx)
            model.train()
            predictions = model(data)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict(y_train):
    data = y_train
    pred = []
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        predictions = model(data)
        _, predictions = predictions.max(1)
    return predictions

def evaluate(predictions, targets):
    n_correct = 0
    # print(targets.shape, predictions.shape)
    # print(type(targets), targets.shape, type(predictions), len(predictions))
    n_samples = targets.size(0)
    n_correct += (predictions == targets).sum().item()
    acc = n_correct / n_samples
    print(f'accuracy = {acc}')
    return acc

from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, features, labels):
        super(Dataset, self).__init__()
        self.features = features
        self.labels = labels
        self.shape = features.shape, labels.shape
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def binary_confusion(y_pred, target):
    # matrix represents [pred 0 targ 0, pred 0 targ 1] [pred 1 targ 0, pred 1 targ 1]
    matrix = [[0, 0],[0, 0]]
    for i in range(y_pred.shape[0]):
        #zero stands for not thumb
        if y_pred[i] == 0 and target[i] == 0:
            matrix[0][0] += 1
        elif y_pred[i] == 0 and target[i] == 1:
            matrix[0][1] += 1
        elif y_pred[i] == 1 and target[i] == 0:
            matrix[1][0] += 1
        elif y_pred[i] == 1 and target[i] == 1:
            matrix[1][1] += 1
    matrix = matrix/np.sum(matrix)
    return matrix

# load from .pkl files ("raw")
def load_data():
    raw_features = {}
    raw_labels = {}
    fileList = []

    for letter in ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I']: 
        raw_features[f'subj_{letter}'] = np.load(f'pickles/subj_{letter}_features{affix}.npy')
        raw_labels[f'subj_{letter}'] = np.load(f'pickles/subj_{letter}_labels{affix}.npy').reshape(-1)
        fileList.append((f'subj_{letter}', raw_features[f'subj_{letter}'], raw_labels[f'subj_{letter}']))
    return fileList

# original 2 class classifier code --> w/ the helper functions :D


fileList = load_data()

batch_size = 32
dropout = 0.2
n_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-6
input_shape = (-1, 1, ch, t)

torch.cuda.set_device(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sm = SMOTE()

# train neural net
accuracies = []
for subj, features, labels in fileList:
    finger_predictions = []
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size= 0.3) # 70% training 30% test
    for finger in range(5):
        y_train_ = np.where(y_train == finger, 1, 0)
        y_test_ = np.where(y_test == finger, 1, 0)
        X_res = X_train.reshape((X_train.shape[0], -1))
        X_res, y_res = sm.fit_resample(X_res, y_train_)
        X_res = X_res.reshape((X_res.shape[0], ch, -1))

        model = ConvNet_Finger().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train the neural network

        model.train()
        data = X_res
        targets = y_res
        data = torch.tensor(data.reshape(int(data.shape[0]), 1, data.shape[1], data.shape[2]), dtype=torch.float32) # reshape # of trial, 1 channel, # of samples
        data = data.to(device)
        targets = torch.tensor(targets, dtype=torch.int64)
        targets = targets.to(device)
        res_data = EEGDataset(data, targets)
        res_loader = DataLoader(res_data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(50):
            # print("epoch = ",epoch)
            model.train()

            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # Evaluate the neural network
        pred = predict(torch.tensor(X_test.reshape(input_shape), dtype=torch.float32).to(device))
        y_res = np.where(y_test == finger, 1, 0)
        y_res = torch.tensor(y_res, dtype=torch.int64).to(device)
        acc = evaluate(pred, y_res)
        print(f"test accuracy {subj} {finger} = ", acc)
        print(f'confusion matrix {binary_confusion(pred, y_res)}')

    print(f'avg accuracy = {np.mean(accuracies)}')

