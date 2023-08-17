import numpy as np
import matplotlib.pyplot as plt
from utils import generate_clusterization_data, split_data, accuracy



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
    generated_data, generated_labels = generate_clusterization_data(n_clusters = 2, n_samples = 300)

    generated_labels = generated_labels * 2 - 1 #normalize labels to [-1; 1]
    x_train, x_test, y_train, y_test =  split_data(generated_data, generated_labels, ratio = 0.25)


    svm = SVM()
    svm.fit(x_train, y_train)

    w = svm.w
    b = svm.b

    # x * w0 + y * w1 - b = 0 => y = -(x * w0 - b) / w1
    y = lambda x: -(x * w[0] - b) / w[1]

    x_disp = np.linspace(np.min(x_train[:,0]), np.max(x_train[:,0]), num=10)
    y_disp = [y(x) for x in x_disp]

    plt.title("Support Vector Machine")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.scatter(x_train[y_train == 1][:,0], x_train[y_train == 1][:,1], marker='_',color='blue', label='cluster 1')
    plt.scatter(x_train[y_train == -1][:,0], x_train[y_train == -1][:,1], marker='+',color='green',  label='cluster 2')
    
    plt.plot(x_disp, y_disp, 'red', label='SVM')

    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()
        


