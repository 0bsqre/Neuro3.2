import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self


class MLP:
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes=5):
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),
            np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))
        ]
        self.learning_rate = learning_rate
        self.layers = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
     
    def feed_forward(self, x):
        input_ = x
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0]))
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))
        
        self.layers = [input_, hidden_, output_]
        return self.layers[-1]
    
    def backward(self, target):
        err = (target - self.layers[-1])
        
        for i in range(len(self.layers)-1, 0, -1):
            err_delta = err * self.derivative_sigmoid(self.layers[i])
            err = np.dot(err_delta, self.weights[i - 1].T)
            dw = np.dot(self.layers[i - 1].T, err_delta)
            self.weights[i - 1] += self.learning_rate * dw
            
    def train(self, x_values, target):
        for xi, yi in zip(x_values, target):
            xi = xi.reshape(1, -1)
            yi = yi.reshape(1, -1)
            self.feed_forward(xi)
            self.backward(yi)
    
    def predict(self, x_values):
        return self.feed_forward(x_values)