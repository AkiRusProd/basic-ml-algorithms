import numpy as np
import matplotlib.pyplot as plt
from decision_tree_regressor import DecisionTreeRegressor


def generate_dataset(n = 30, beta = 10, variance_reduction = 10):

    e = (np.random.randn(n) * variance_reduction).round(decimals = 1)

    x = (np.random.rand(n) * n)
    y = (np.random.rand(n) * n)

    z = x * beta + y * beta + e
    x, y, z = np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1), np.expand_dims(z, axis = 1)

    return np.concatenate((x, y, z), axis=1)
 




def split_data(data, ratio):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data = data[indices].reshape(data.shape)

    train_data, test_data = data[:int(len(data) * (1 - ratio))], data[-int(len(data) * ratio):]
   
    return train_data[:, :2], test_data[:, :2], train_data[:, 2], test_data[:, 2]


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
    data = generate_dataset(200)
    x_train, x_test, y_train, y_test = split_data(data, ratio = 0.25)


    rfc = RandomForestRegressor()

    rfc.fit(x_train, y_train)
    y_predicted = rfc.predict(x_test)


    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, 
                label ='train values', s = 5, color ="dodgerblue")

    ax.scatter(x_test[:, 0], x_test[:, 1], y_test,
                    label ='test values', s = 5, color ="blue")
    
    ax.scatter(x_test[:, 0], x_test[:, 1], y_predicted,
                    label ='predicted values', s = 5, color ="orange")
    ax.legend()
    ax.view_init(45, 0)
    
    plt.show()
