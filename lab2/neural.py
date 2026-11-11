import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.layerSizes = [inputSize] + hiddenSizes + [outputSize]
        self.W = []
        
        for i in range(len(self.layerSizes) - 1):
            prevSize = self.layerSizes[i]
            currSize = self.layerSizes[i + 1]
            W_layer = np.zeros((1 + prevSize, currSize))
            W_layer[0, :] = np.random.randint(0, 3, size=currSize)
            W_layer[1:, :] = np.random.randint(-1, 2, size=(prevSize, currSize))
            self.W.append(W_layer)

    def predict(self, Xp):
        self.activations = [Xp]
        current = Xp
        
        for i, W_layer in enumerate(self.W):
            z = np.dot(current, W_layer[1:, :]) + W_layer[0, :]
            
            current = np.where(z >= 0.0, 1, -1).astype(np.float64)
            self.activations.append(current)
        
        out = current
        hidden = self.activations[1:-1]  # все скрытые слои
        return out, hidden

    def train(self, X, y, n_iter=5, eta=0.01):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        for i in range(n_iter):
            print(self.W[-1].reshape(1, -1))  # вывод весов выходного слоя
            for xi, target in zip(X, y):
                xi = xi.reshape(1, -1)
                target = target.reshape(1, -1)
                
                pr, _ = self.predict(xi)
                
                # Обновление всех слоёв, начиная с выходного
                error = target - pr
                
                for layer_idx in range(len(self.W) - 1, -1, -1):
                    hidden_input = self.activations[layer_idx]  # вход для этого слоя
                    self.W[layer_idx][1:, :] += eta * error * hidden_input.T
                    self.W[layer_idx][0, :] += eta * error.flatten()
                    
                    # Подготовка ошибки для предыдущего слоя
                    if layer_idx > 0:
                        error = np.dot(error, self.W[layer_idx][1:,:].T)
        return self

