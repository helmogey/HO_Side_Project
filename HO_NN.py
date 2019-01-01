

import pandas as pd
import numpy as np
from torch import nn
from torch import optim
import torch
from random import shuffle
import time
from torch.optim import lr_scheduler
import copy




X_train = pd.read_csv('x_train.csv')
Y_train = pd.read_csv('y_train.csv')

X_test = pd.read_csv('x_test.csv')
Y_test = pd.read_csv('y_test.csv')




X_train['RSRP_S'] = (X_train['RSRP_S'] - np.mean(X_train['RSRP_S'])) / (np.max(X_train['RSRP_S']) - np.min(X_train['RSRP_S']))
X_train['RSRP_t'] = (X_train['RSRP_t'] - np.mean(X_train['RSRP_t'])) / (np.max(X_train['RSRP_t']) - np.min(X_train['RSRP_t']))
X_train['Tr'] = (X_train['Tr'] - np.mean(X_train['Tr'])) / (np.max(X_train['Tr']) - np.min(X_train['Tr']))


X_test['RSRP_S'] = (X_test['RSRP_S'] - np.mean(X_train['RSRP_S'])) / (np.max(X_train['RSRP_S']) - np.min(X_train['RSRP_S']))
X_test['RSRP_t'] = (X_test['RSRP_t'] - np.mean(X_train['RSRP_t'])) / (np.max(X_train['RSRP_t']) - np.min(X_train['RSRP_t']))
X_test['Tr'] = (X_test['Tr'] - np.mean(X_train['Tr'])) / (np.max(X_train['Tr']) - np.min(X_train['Tr']))



one_hot_data = pd.concat([X_train, pd.get_dummies(X_train['S'], prefix='S')], axis=1)
one_hot_data = one_hot_data.drop('S', axis=1)

one_hot_data = pd.concat([one_hot_data, pd.get_dummies(X_train['T'], prefix='T')], axis=1)
one_hot_data = one_hot_data.drop('T', axis=1)


X_data = np.array(one_hot_data)
Y_data = np.array(Y_train)




def shuflle_(x,y,valid):
    num_train = len(x)
    idx = list(range(num_train))
    t = int(((100-valid)*num_train)/100)
    shuffle(idx)
    idx_train = idx[0:t]
    idx_valid = idx[t+1:]

    x_train = x[idx_train,:]
    y_train = y[idx_train,:]

    x_valid = x[idx_valid, :]
    y_valid = y[idx_valid, :]

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train)

    x_valid = torch.from_numpy(x_valid).float()
    y_valid = torch.from_numpy(y_valid)
    X = {}
    X["train"] = x_train
    X["val"] = x_valid

    Y = {}
    Y["train"] = y_train
    Y["val"] = y_valid

    return X,Y




class Network(nn.Module):
    def __init__(self,h1,h2,drop):
        super().__init__()

        self.h1 = nn.Linear(101, h1)
        self.h2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 2)

        self.RelU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.h1(x)
        x = self.dropout(self.RelU(x))
        x = self.h2(x)
        x = self.dropout(self.RelU(x))
        x = self.output(x)
        x = self.softmax(x)

        return x










def train_model(model, criterion, optimizer, scheduler, valid=20,num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        X, Y = shuflle_(X_data, Y_data, valid)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for featcher, labels in zip(X[phase], Y[phase]):

                featcher = featcher.view(featcher.shape[0], -1)
                featcher = torch.transpose(featcher, 0, 1)

                featcher = featcher.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(featcher)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * featcher.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(X[phase])
            epoch_acc = running_corrects.double() / len(X[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'HO.pt')
                print("Saving model>>>>>>")
            else:
                print('last Best val Acc: {:4f}'.format(best_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model



device = "cuda"
h1 = 128
h2 = 64
drop = 0.2
valid = 20
epochs = 50
lr = 0.01
moment = 0.9
gamma = 0.1
step_size = 7

model = Network(h1,h2,drop)
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr,momentum=moment)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)



model = train_model(model, criterion, optimizer, exp_lr_scheduler,valid,epochs)


