import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate_clusterization_data, split_data
from metrics import accuracy, roc_сurve, pr_curve, auc


# https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient

class Sigmoid():
    def __call__(self, x):
        return self.func(x)
    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        f_x = self.func(x)
        return f_x * (1.0 - f_x)
    

class BCE:
    def __call__(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def grad(self, y_pred, y_true):
        return - (y_true/(y_pred + 1E-15) - (1 - y_true)/(1 - y_pred + 1E-15))


class LogisticRegression():
    def __init__(self, n_iterations = 1000, lr = 0.01):
        self.n_iterations = n_iterations
        self.lr = lr

        self.weight = None
        self.bias = None

        self.activation = Sigmoid()
        self.loss_fn = BCE()

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
        self.init_weights(n_features = n_features)

        losses = []
        tqdm_range = tqdm(range(self.n_iterations), total = self.n_iterations)
        for i in range(self.n_iterations):
            tqdm_range.update(1)
            for x_true, y_true in zip(X, y):
                y_true = y_true[:, None]
                x_true = x_true[None, ...] if x_true.ndim == 1 else x_true

                z = np.matmul(x_true, self.weight.T) + self.bias
                y_pred = self.activation(z)
                
                # grad = dBinaryLogLoss/dw = -dL(p, y)/dp * dp/dz * dz/dw = (p - y) * x; where p = F(z) = Sigmoid(z)
                # So you can do the math something like this:
                # grad = self.loss_fn.grad(y_pred, y_true) * self.activation.grad(z) #equals -dL(p, y)/dp * dp/dz = p - y
                # Or simply like this:
                grad = y_pred - y_true
                
                self.weight -= self.lr * np.matmul(x_true.T, grad).T #np.matmul(grad.T, x_true) equals dz/dw
                self.bias -= self.lr * np.sum(grad, axis = 0)


                loss = self.loss_fn(y_pred, y_true)

                tqdm_range.set_description(f'epoch: {i + 1}/{self.n_iterations}, loss: {loss:.7f}')
                losses.append(loss)

        return losses
	
    def predict(self, X):
        return self.activation(np.dot(X, self.weight.T) + self.bias)





if __name__ == '__main__':
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 100)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)

    model = LogisticRegression(n_iterations=1000)
    losses = model.fit(X_train, y_train[:, None])
    y_score = model.predict(X_test)
    
    y_pred = np.where(y_score >= 0.5, 1, 0)

    print(f"accuracy: {accuracy(y_test, y_pred[...,0]) * 100}%")

    w = model.weight
    b = model.bias

    x_disp = np.linspace(np.min(X_test[:,0]), np.max(X_test[:,0]), num=10)
   
    # x * w0 + y * w1 + b = 0 => y = -(x * w0 + b) / w1 (separating hyperplane)
    # you can verify this if you express z from this expression: sigmoid(z) = 0.5, where z = Σ(xi * wi) + b; so z = Σ(xi * wi) + b = 0
    y = lambda x: -(x * w[0][0] + b[0][0]) / w[0][1]
    y_disp= [y(x) for x in x_disp]

    # plot Classification decision boundary
    plt.title("Logistic Regression")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.scatter(X_test[y_test == 1][:,0], X_test[y_test == 1][:,1], marker='_',color='blue', label='cluster 1')
    plt.scatter(X_test[y_test == 0][:,0], X_test[y_test == 0][:,1], marker='+',color='green',  label='cluster 2')
    
    plt.plot(x_disp, y_disp, 'red', label='Gradient descent') 

    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()

    fpr, tpr, thresholds = roc_сurve(y_test, y_score.squeeze(1))
    print(f"AUC: {auc(fpr, tpr)}")
 
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()

    precision, recall, thresholds = pr_curve(y_test, y_score.squeeze(1))
    print(f"AUC: {auc(recall, precision)}")

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.show()