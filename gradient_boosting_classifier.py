import numpy as np 

from decision_tree_regressor import DecisionTreeRegressor, Node
from utils import generate_clusterization_data, split_data
from metrics import accuracy

#https://maelfabien.github.io/machinelearning/GradientBoostC/#gradient-boosting-classification-steps
#https://www.youtube.com/watch?v=jxuNLH5dXCs&ab_channel=StatQuestwithJoshStarmer
#https://www.youtube.com/watch?v=StWY5QWMXCw&ab_channel=StatQuestwithJoshStarmer
#https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient


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
            

        leaf_value = np.sum(preds) / np.sum(prev_probs * (1 - prev_probs)) #transformation; (prev_probs * (1 - prev_probs) is Hessian matrix)
    
        return Node(class_value = leaf_value)

    def fit(self, samples, preds, prev_probs):
        self.features_num = samples.shape[1]

        if self.criterion == 'mse':
            self.criterion_func = self.compute_mse
        elif self.criterion == 'mae':
            self.criterion_func = self.compute_mae
        elif self.criterion == 'variance reduction':
            self.criterion_func = self.compute_variance_reduction
        else:
            raise SystemExit(f'Criterion with name "{self.criterion}" not found') 

        self.tree = self.insert_tree(data = np.concatenate((samples, np.array(prev_probs, ndmin = 2).T, np.array(preds, ndmin = 2).T,), axis = 1))
        

class BinaryGradientBoostingClassifier():
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
            #antigrad = -grad =  -dBinaryLogLoss(y_i, F(x_i))/dx_i = - -(y_i - F(x_i)) = y_i - F(x_i) where F(X) is Sigmoid
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




class MulticlassGradientBoostingClassifier():
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split = 2, min_samples_leaf = 2, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.estimators = []
    
    def softmax(self, x):
        e_x =  np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def one_hot_encode(self, y):
        unique_labels = np.unique(y)
        encoded_labels = np.zeros((len(y), len(unique_labels)), dtype=int)
        for idx, label in enumerate(unique_labels):
            encoded_labels[y == label, idx] =  1
        return encoded_labels
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        
        y = self.one_hot_encode(y)
        y_pred = np.zeros(shape=y.shape)

        for _ in range(self.n_estimators):
            class_estimators = []
            for c in range(self.n_classes):
                #antigrad = -grad  -dLogLoss(y_i, F(x_i))/dx_i = - -(y_i - F(x_i)) = y_i - F(x_i) where F(X) is Softmax
                residuals = y[:, c] - self.softmax(y_pred)[:, c] #residual = observed - predicted
                #fit a new base model on the residuals
                estimator = DecisionTreeRegressorClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
                estimator.fit(X, residuals, self.softmax(y_pred)[:, c])
                #update predictions
                y_pred[:, c] += self.learning_rate * np.array(estimator.predict(X))
                #add the new model to the list
                class_estimators.append(estimator)
            self.estimators.append(class_estimators)

    def predict(self, X):
        y_pred = np.zeros((len(X), self.n_classes))

        for c in range(self.n_classes):
            for estimator in self.estimators:
                y_pred[:, c] += self.learning_rate * np.array(estimator[c].predict(X))
        
        y_pred = self.softmax(y_pred)

        return np.argmax(y_pred, axis=1)



if __name__ == "__main__":
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 300)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)
   
    gbc = BinaryGradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=2)

    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)

    print(f"binary gbc accuracy: {accuracy(y_test, y_pred) * 100}%")


    X_train, y_train = generate_clusterization_data(n_clusters = 3, n_samples = 300)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)
   
    gbc = MulticlassGradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=2)

    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)

    print(f"multiclass gbc accuracy: {accuracy(y_test, y_pred) * 100}%")

