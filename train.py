from ctypes import sizeof
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import TensorDataset

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# first process data
df = pd.read_csv("./data/CH22_Demand_XY_Train.csv")
df_X = df.iloc[:, 1:5]
df_y = df.iloc[:, 5]

X = df_X.to_numpy()
y = df_y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# load the model
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs

epochs = 200
input_dim = 4
output_dim = 1
lr_rate = 0.001

model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

iter = 0
for epoch in range(int(epochs)):
    for i, (inputs, outputs) in enumerate(train_loader):
        # print("epoch: {}, i: {}".format(epoch, i))
        inputs = Variable(inputs.view(-1, 4))
        outputs = Variable(outputs.view(-1, 1))

        optimizer.zero_grad()
        predictions = model(inputs)
        # loss = criterion(predictions, outputs)
        loss = r2_loss(predictions, outputs)
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%500==0:
            # calculate Accuracy
            # correct = 0
            # total = 0
            for inputs, outputs in test_loader:
                inputs = Variable(inputs.view(-1, 4))
                outputs = Variable(outputs.view(-1, 1))
                predictions = model(inputs)
                # _, predicted = torch.max(predictions.data, 1)
                # total+= outputs.size(0)
                test_loss = r2_loss(predictions, outputs)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                # correct+= (predicted == outputs).sum()
            # accuracy = 100 * correct/total
            # print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
            print("Iteration: {}. Loss: {}.".format(iter, test_loss))


