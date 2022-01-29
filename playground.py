from ctypes import sizeof
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from sklearn.linear_model import LinearRegression

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(r2)
    return r2

# first process data
df = pd.read_csv("./data/CH22_Demand_XY_Train.csv")
df_X = df.iloc[:, 1:5]
df_y = df.iloc[:, 5]

X = df_X.to_numpy()
y = df_y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('my total loss: {}'.format(r2_loss(torch.Tensor(y), torch.Tensor(y))))