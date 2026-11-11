import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('dataset_simple.csv')

# Подготовка данных
X = df.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)  # Возраст (2D массив)
y = df.iloc[:, 1].values.astype(np.float32)  # Доход

# Преобразование в тензоры PyTorch
X_tensor = torch.from_numpy(X)  # Конвертируем numpy в torch.Tensor
y_tensor = torch.from_numpy(y)

# Визуализация
plt.figure()
plt.scatter(X, y, marker='o')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income')
plt.show()

X = (X - X.mean()) / X.std()  # Нормализация возраста
y = (y - y.mean()) / y.std()  # Нормализация дохода


# Определение модели
class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
# Параметры модели
inputSize = 1  # Один признак - возраст
hiddenSizes = 64
outputSize = 1


# Инициализация модели
net = NNet_regression(inputSize, hiddenSizes, outputSize)

# Функция потерь и оптимизатор
lossFn = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# Обучение
epochs = 1000
for i in range(epochs):
    # Передаем тензор, а не numpy-массив
    pred = net(X_tensor)
    loss = lossFn(pred.squeeze(), y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f'Эпоха {i+1}, Ошибка: {loss.item():.2f}')
  

# Оценка
with torch.no_grad():
    pred = net(X_tensor)

print('\nПервые 10 предсказаний:')
print(pred[:10].numpy().flatten())

err = torch.mean(torch.abs(y_tensor - pred.squeeze()))
print(f'\nСредняя абсолютная ошибка (MAE): {err.item():.2f}')


























