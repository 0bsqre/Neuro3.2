# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from neural import Perceptron

df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1]
hiddenSizes = [10, 5]
outputSize = 1

NN = Perceptron(inputSize, hiddenSizes, outputSize)
NN.train(X, y, n_iter=5, eta=0.01)

y_test = df.iloc[:, 4].values
y_test = np.where(y_test == "Iris-setosa", 1, -1)
X_test = df.iloc[:, [0, 2]].values

out, hidden_predict = NN.predict(X_test)
print("Сумма ошибок:", np.sum(out - y_test.reshape(-1, 1)))
