import numpy as np
import matplotlib.pyplot as plt
from decision_tree_regressor import DecisionTreeRegressor
from utils import generate_regression_data, split_data


#https://en.wikipedia.org/wiki/Random_forest

class RandomForestRegressor():

    def __init__(self, estimators_num = 100, min_samples_split = 2, min_samples_leaf = 2, max_depth = 2):

        self.estimators = None

        self.estimators_num = estimators_num

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        

    def simulate_data(self, data):
        simulated_data = []

        for _ in range(self.estimators_num):
            simulated_data.append(data[np.random.choice(len(data), size = len(data), replace = True), :])
        
            
        return np.asfarray(simulated_data)

    def create_estimators(self):
        estimators = []

        for _ in range(self.estimators_num):
            estimators.append(DecisionTreeRegressor(self.min_samples_split, self.min_samples_leaf, self.max_depth))

        return estimators

    def fit_estimators(self, data):

        for i in range(self.estimators_num):
            self.estimators[i].fit(data[i, :, : -1], data[i, :, -1])

    
    def predict_estimators(self, data):
        estimators_predictions = []

        for i in range(self.estimators_num):
            estimators_predictions.append(self.estimators[i].predict(data))

        return np.asfarray(estimators_predictions)
     
    
    def fit(self, x, y):
        
        simulated_data = self.simulate_data(np.concatenate((x, np.array(y, ndmin = 2).T), axis = 1))
        self.estimators = self.create_estimators()

        self.fit_estimators(simulated_data)


    def predict(self, x):
        
        estimators_predictions = self.predict_estimators(x)

        predictions = [np.mean(column) for column in estimators_predictions.T]
        
        return predictions




if __name__ == "__main__":
    X_train, y_train = generate_regression_data(100)
    X_train, X_test, y_train, y_test = split_data(X_train, y_train, ratio = 0.25)

    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train[:, 0])
    y_pred = rfr.predict(X_test)

    indices = np.argsort(X_test[:, 0])

    xs = np.array(X_test)[indices]
    ys = np.array(y_pred)[indices]
    
    f = plt.figure(figsize = (16 * 0.5, 9 * 0.5))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(X_test, y_test, 'o')
    ax.plot(xs, ys, 'r')
    ax.set_title('Random Forrest Regressor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
   
    plt.grid()
    plt.show()
