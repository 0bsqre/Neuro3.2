# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from neural import MLP

df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1, 1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1]
hiddenSizes = 10
outputSize = 1

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

for i in range(iterations):
    net.train(X, y)
    
    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + 
              "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

pr = net.predict(X)
print("Ошибка классификации:", sum(abs(y - (pr > 0.5))))