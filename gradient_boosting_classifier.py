import numpy as np 

from decision_tree_regressor import DecisionTreeRegressor, Node
from utils import generate_clusterization_data, split_data, accuracy

#https://maelfabien.github.io/machinelearning/GradientBoostC/#gradient-boosting-classification-steps
#https://www.youtube.com/watch?v=jxuNLH5dXCs&ab_channel=StatQuestwithJoshStarmer
#https://www.youtube.com/watch?v=StWY5QWMXCw&ab_channel=StatQuestwithJoshStarmer


class DecisionTreeRegressorClassifier(DecisionTreeRegressor):
    """Gradient Boosting Decision Tree for Classification based on Regression"""
    def __init__(self, min_samples_split=2, min_samples_leaf=2, max_depth=5):
        super().__init__(min_samples_split, min_samples_leaf, max_depth)

    def insert_tree(self, data, tree_depth = 0): 
        preds = data[:,-1]
        prev_probs = data[:,-2]

        samples_num = len(data)

        if samples_num >= self.min_samples_split and tree_depth <= self.max_depth:

            left_data, right_data, feature_index, threshold_value, information_gain = self.find_best_split(data)

            if len(left_data) >= self.min_samples_leaf and len(right_data) >= self.min_samples_leaf:
         
                if information_gain > 0:
                    
                    left_subtree = self.insert_tree(left_data, tree_depth+1)
                    
                    right_subtree = self.insert_tree(right_data, tree_depth+1)
                    
                    return Node(feature_index, threshold_value, 
                                left_subtree, right_subtree, information_gain)
            

        leaf_value = np.sum(preds) / np.sum(prev_probs * (1 - prev_probs)) #transformation
    
        return Node(class_value = leaf_value)

    def fit(self, samples, preds, prev_probs):
        self.features_num = samples.shape[1]

        self.tree = self.insert_tree(data = np.concatenate((samples, np.array(prev_probs, ndmin = 2).T, np.array(preds, ndmin = 2).T,), axis = 1))
        

class GradientBoostingClassifier():
    """Gradient Boosting Binary Classifier"""
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split = 2, min_samples_leaf = 2, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.estimators = []

    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))
    
    def fit(self, X, y):
        log_odds = np.log(len(y[y==1])/len(y[y==0]))
        
        y_pred = np.full(len(X), log_odds)
        self.init_y_pred = log_odds

        for _ in range(self.n_estimators):
            #antigrad = -grad =  -dBinaryLogLoss(y_i, F(x_i))/dF(x_i) = - -(y_i - F(x_i)) = y_i - F(x_i)
            residuals = y - self.sigmoid(y_pred) #residual = observed - predicted
            #fit a new base model on the residuals
            estimator = DecisionTreeRegressorClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            estimator.fit(X, residuals, self.sigmoid(y_pred))
            #update predictions
            y_pred += self.learning_rate * np.array(estimator.predict(X))
            #add the new model to the list
            self.estimators.append(estimator)

    def predict(self, X):
        y_pred = np.full(len(X), self.init_y_pred)

        for estimator in self.estimators:
            y_pred += self.learning_rate * np.array(estimator.predict(X))
        
        y_pred = self.sigmoid(y_pred)

        return np.where(y_pred >= 0.5, 1, 0)


if __name__ == "__main__":
    generated_data, generated_labels = generate_clusterization_data(n_clusters = 2, n_samples=300)
    x_train, x_test, y_train, y_test =  split_data(generated_data, generated_labels, ratio = 0.25)
   
    gbc = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=2)

    gbc.fit(x_train, y_train)
    y_pred = gbc.predict(x_train)

    print(f"accuracy: {accuracy(y_train, y_pred) * 100}%")
