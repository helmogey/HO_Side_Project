

import matplotlib
matplotlib.use('Qt5Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torch



X_train = pd.read_csv('x_train.csv')
Y_train = pd.read_csv('y_train.csv')

X_test = pd.read_csv('x_test.csv')
Y_test = pd.read_csv('y_test.csv')

Mean_RSRP_S = np.mean(X_train['RSRP_S'])
Max_RSRP_S = np.max(X_train['RSRP_S'])
Min_RSRP_S = np.min(X_train['RSRP_S'])

Mean_RSRP_t = np.mean(X_train['RSRP_t'])
Max_RSRP_t = np.max(X_train['RSRP_t'])
Min_RSRP_t = np.min(X_train['RSRP_t'])

Mean_Tr = np.mean(X_train['Tr'])
Max_Tr = np.max(X_train['Tr'])
Min_Tr = np.min(X_train['Tr'])

X_train['RSRP_S'] = (X_train['RSRP_S'] - Mean_RSRP_S) / (Max_RSRP_S - Min_RSRP_S)
X_train['RSRP_t'] = (X_train['RSRP_t'] - Mean_RSRP_t) / (Max_RSRP_t - Min_RSRP_t)
X_train['Tr'] = (X_train['Tr'] - Mean_Tr) / (Max_Tr - Min_Tr)

X_test['RSRP_S'] = (X_test['RSRP_S'] - Mean_RSRP_S) / (Max_RSRP_S - Min_RSRP_S)
X_test['RSRP_t'] = (X_test['RSRP_t'] - Mean_RSRP_t) / (Max_RSRP_t - Min_RSRP_t)
X_test['Tr'] = (X_test['Tr'] - Mean_Tr) / (Max_Tr - Min_Tr)



one_hot_data = pd.concat([X_train, pd.get_dummies(X_train['S'], prefix='S')], axis=1)
one_hot_data = one_hot_data.drop('S', axis=1)

one_hot_data = pd.concat([one_hot_data, pd.get_dummies(X_train['T'], prefix='T')], axis=1)
one_hot_data = one_hot_data.drop('T', axis=1)


X_train = np.array(one_hot_data)
Y_train = np.array(Y_train)


one_hot_data = pd.concat([X_test, pd.get_dummies(X_test['S'], prefix='S')], axis=1)
one_hot_data = one_hot_data.drop('S', axis=1)

one_hot_data = pd.concat([one_hot_data, pd.get_dummies(X_test['T'], prefix='T')], axis=1)
one_hot_data = one_hot_data.drop('T', axis=1)


X_test = np.array(one_hot_data)
Y_test = np.array(Y_test)

# print(X_test)
device = "cuda"

X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train)

X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test)








class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(101, 128)
        self.h2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)

        self.RelU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = self.h1(x)
        x = self.dropout(self.RelU(x))
        x = self.h2(x)
        x = self.dropout(self.RelU(x))
        x = self.output(x)
        x = self.softmax(x)

        return x


model = Network()




model.cuda()


criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load("HO128.pt"))

test_loss = 0.0
correct = 0

model.eval()

for data, target in zip(X_test, Y_test):

    data = data.view(data.shape[0], -1)
    data = torch.transpose(data, 0, 1)
    data, target = data.cuda(), target.cuda()

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)

    # print(output)
    _, pred = torch.max(output, 1)
    # print(pred)
    # print(target)

    correct += torch.sum(pred == target.data)


test_loss = test_loss/len(X_test)
print('Test Loss: {:.6f}\n'.format(test_loss))


correct = int(correct)/len(X_test)*100

print('Accuracy: {:.6f}\n'.format(correct))






