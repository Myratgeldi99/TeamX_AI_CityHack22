from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

###

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



def loader(file_name="../data/CH22_Demand_XY_Train.csv", degree=10):
    df = pd.read_csv(file_name)
    df_X = df.iloc[:, 1:5]
    df_y = df.iloc[:, 5]

    X = df_X.to_numpy()
    y = df_y.to_numpy()

    # poly = PolynomialFeatures(degree)
    # poly.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = loader()
    model = Pipeline([('poly', PF(degree=12)),
                   ('linear', LinearRegression(fit_intercept=False))])
    
    model = model.fit(X_train, y_train)

    print(model.score(X_test, y_test))