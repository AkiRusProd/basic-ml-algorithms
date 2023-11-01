import numpy as np
import matplotlib.pyplot as plt
from utils import generate_clusterization_data, split_data
from metrics import accuracy



#https://en.wikipedia.org/wiki/Support_vector_machine


class SVM:
    def __init__(self, lr =0.001, lambda_ = 0.01, n_iterations=1000):
        self.lr = lr
        self.lambda_ = lambda_
        self.n_iterations = n_iterations

        self.w = None
        self.b = None

    def fit(self, X, y):
        assert np.max(y) == 1 and np.min(y) == -1, "Only binary classification is supported"
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features) if self.w is None else self.w
        self.b = 0 if self.b is None else self.b

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                #Soft margin using Hinge Loss with L2 Regularization
                #L(w) = Σ max(0, 1 - y * (np.dot(X[i], self.w) - self.b)) + λ||w||^2
                margin = y[i] * (np.dot(X[i], self.w) - self.b)
                if margin >= 1:
                    self.w -= self.lr * (2 * self.lambda_ * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_ * self.w - X[i] * y[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)


if __name__ == "__main__":
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 300)

    y_train = y_train * 2 - 1 #normalize labels to [-1; 1]
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)


    svm = SVM()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f"accuracy: {accuracy(y_test, y_pred) * 100}%")

    w = svm.w
    b = svm.b

    x_disp = np.linspace(np.min(X_test[:,0]), np.max(X_test[:,0]), num=10)

    #optimal hyperplane
    #x * w0 + y * w1 - b = 0 
    #express y from equation
    # => y = -(x * w0 - b) / w1
    y = lambda x: -(x * w[0] - b) / w[1]
    y_disp = [y(x) for x in x_disp]

    plt.plot(x_disp, y_disp, 'red', label='SVM')

    #first edge of the hyperplane
    #x * w0 + y * w1 - b = 1
    #express y from equation
    # => y = -(x * w0 - 1 - b) / w1
    y = lambda x: -(x * w[0] - 1 - b) / w[1]
    y_disp = [y(x) for x in x_disp]

    plt.plot(x_disp, y_disp, 'red', label='edge', linestyle=':', linewidth=0.5)

    #second edge of the hyperplane
    #x * w0 + y * w1 - b = -1
    #express y from equation
    # => y = -(x * w0 + 1 - b) / w1
    y = lambda x: -(x * w[0] + 1 - b) / w[1]
    y_disp = [y(x) for x in x_disp]

    plt.plot(x_disp, y_disp, 'red', label='edge', linestyle=':', linewidth=0.5)

    #plot Classification decision boundary
    plt.title("Support Vector Machine")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.scatter(X_test[y_test == 1][:,0], X_test[y_test == 1][:,1], marker='_',color='blue', label='cluster 1')
    plt.scatter(X_test[y_test == -1][:,0], X_test[y_test == -1][:,1], marker='+',color='green',  label='cluster 2')

    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()
        


