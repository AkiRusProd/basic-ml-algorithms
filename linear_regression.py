import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from  utils import generate_linear_regression_data, split_data


class MSE:
    def __call__(self, y_pred, y_true):
        return np.sum((y_true - y_pred) ** 2) / y_true.size
    
    def grad(self, y_pred, y_true):
        return -2 * (y_true - y_pred) / y_true.size





class SGDRegressor:
    def __init__ (self, n_iterations = 100, lr = 0.0001):
        self.n_iterations = n_iterations
        self.lr = lr

        self.weight = None
        self.bias = None

        self.loss_fn = MSE()

    def init_weights(self, n_features):
        if self.weight is None or self.bias is None:
            # self.weight = np.random.uniform(-1, 1, (1, n_features)) if self.weight is None else self.weight
            # self.weight = np.random.normal(0, pow(n_features, -0.5), (1, n_features)) if self.weight is None else self.weight
            # self.weight = np.random.normal(0, 1, (1, n_features)) if self.weight is None else self.weight

            # Xavier initialization
            stdv = 1 / np.sqrt(n_features)
            self.weight = np.random.uniform(-stdv ,stdv,  (1, n_features)) if self.weight is None else self.weight
                        
            self.bias = np.zeros((1, 1)) if self.bias is None else self.bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.init_weights(n_features)

        losses = []
        tqdm_range = tqdm(range(self.n_iterations), total = self.n_iterations)
        for i in range(self.n_iterations):
            tqdm_range.update(1)
            for x_true, y_true in zip(X, y):
                y_true = y_true[:, np.newaxis]
                x_true = x_true[None, ...] if x_true.ndim == 1 else x_true

                y_pred = np.matmul(x_true, self.weight) + self.bias

                loss = self.loss_fn(y_pred, y_true)

                grad = self.loss_fn.grad(y_pred, y_true)

                self.weight -= self.lr * np.matmul(grad.T, x_true)
                self.bias -= self.lr * np.sum(grad)
                losses.append(loss)

                tqdm_range.set_description(f'epoch: {i + 1}/{self.n_iterations}, loss: {loss:.7f}')

        return losses
    
    def predict(self, X):
        y_pred = np.matmul(X, self.weight) + self.bias
        return y_pred
				

class OrdinaryLeastSquares:
    def __init__(self) -> None:
        self.b = None

    def add_bias(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)

    def fit(self, X, y):
        X = self.add_bias(X)
        #b* = (X^T * X)^-1 * X^T * y
        self.b = (np.linalg.matrix_power(X.transpose().dot(X), -1)).dot(X.transpose()).dot(y)
        return self.b
    
    def predict(self, X):
        X = self.add_bias(X)
        return X.dot(self.b)
        



if __name__ == '__main__':
    X_train, y_train, true_coefs = generate_linear_regression_data(300)
    X_train, X_test, y_train, y_test = split_data(X_train, y_train, ratio = 0.25)

    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.scatter(X_test, y_test, color ='g', s=10, label='Ground truth') 

    model = SGDRegressor(n_iterations=1000)
    losses = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.plot(X_test, y_pred, 'red', label='Gradient descent') 

    model = OrdinaryLeastSquares()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.plot(X_test, y_pred, 'orange', label='Ordinary least squares') 

    y_true = np.dot(X_test, true_coefs)
    plt.plot(X_test, y_true, 'blue', label='True coefficients')

    plt.legend(loc=2)

    plt.grid(True, linestyle='-', color='0.75')
    plt.show()
