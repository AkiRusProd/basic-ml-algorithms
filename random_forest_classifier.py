import numpy as np
from decision_tree_classifier import DecisionTreeClassifier
from utils import generate_clusterization_data, split_data
from metrics import accuracy


#https://en.wikipedia.org/wiki/Random_forest

class RandomForestClassifier():

    def __init__(self, estimators_num = 100, min_samples_split = 2, min_samples_leaf = 2, max_depth = 2, criterion = 'gini'):

        self.estimators = None

        self.estimators_num = estimators_num

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.criterion = criterion
        

    def simulate_data(self, data):
        simulated_data = []

        for _ in range(self.estimators_num):
            simulated_data.append(data[np.random.choice(len(data), size = len(data), replace = True), :])
        
            
        return np.asfarray(simulated_data)

    def create_estimators(self):
        estimators = []

        for _ in range(self.estimators_num):
            estimators.append(DecisionTreeClassifier(self.min_samples_split, self.min_samples_leaf, self.max_depth, self.criterion))

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

        predictions = [np.argmax(np.bincount(column.astype(int))) for column in estimators_predictions.T]
        
        return predictions





if __name__ == "__main__":
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 300)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)

    rfc = RandomForestClassifier()

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    print(f"accuracy: {accuracy(y_test, y_pred) * 100}%")
