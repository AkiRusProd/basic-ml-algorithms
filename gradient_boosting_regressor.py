import numpy as np 
import matplotlib.pyplot as plt
from decision_tree_regressor import DecisionTreeRegressor
from utils import generate_regression_data, split_data

class GradientBoostingRegressor():
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split = 2, min_samples_leaf = 2, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.estimators = []
    
    def fit(self, X, y):
        y_pred = np.full(len(X), np.mean(y, axis=0))
        self.init_y_pred = np.mean(y, axis=0)

        for _ in range(self.n_estimators):
            #antigrad = -grad =  -dMSE(y_i, F(x_i))/dF(x_i) = - -(y_i - F(x_i)) = y_i - F(x_i)
            residuals = y - y_pred 
            #fit a new base model on the residuals
            estimator = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            estimator.fit(X, residuals)
            #update predictions
            y_pred += self.learning_rate * np.array(estimator.predict(X))
            #add the new model to the list
            self.estimators.append(estimator)

    def predict(self, X):
        y_pred = np.full(len(X), self.init_y_pred)

        for estimator in self.estimators:
            y_pred += self.learning_rate * np.array(estimator.predict(X))

        return y_pred


if __name__ == "__main__":
    X_train, y_train = generate_regression_data(100)
    X_train, X_test, y_train, y_test = split_data(X_train, y_train, ratio = 0.25)

    gbr = GradientBoostingRegressor(n_estimators=30)
    gbr.fit(X_train, y_train[:, 0])
    y_pred = gbr.predict(X_test)

    indices = np.argsort(X_test[:, 0])

    xs = np.array(X_test)[indices]
    ys = np.array(y_pred)[indices]
    
    f = plt.figure(figsize = (16 * 0.5, 9 * 0.5))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(X_test, y_test, 'o')
    ax.plot(xs, ys, 'r')
    ax.set_title('Gradient Boosting Regressor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
   
    plt.grid()
    plt.show()
