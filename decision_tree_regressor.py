import numpy as np 
import matplotlib.pyplot as plt
from utils import generate_regression_data, split_data






"""CART Decision Tree"""
#https://en.wikipedia.org/wiki/Decision_tree_learning

class Node():
    def __init__(self, feature_index=None, threshold_value = None, left = None, right = None, variance_reduction = None, class_value = None):

        self.feature_index = feature_index
        self.threshold_value = threshold_value
        self.left = left
        self.right = right
        self.variance_reduction = variance_reduction
        self.class_value = class_value


class DecisionTreeRegressor():

    def __init__(self, min_samples_split = 2, min_samples_leaf = 2, max_depth = 5):
        self.tree = None
      
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
       


    def find_best_split(self, data):
        max_variance_reduction = -np.inf
        
        best_split_params = [[] for _ in range(5)]

        for feature_index in range(self.features_num):
            for feature_value in np.unique(data[:, feature_index]):
                left_data, right_data = data[data[:, feature_index] <= feature_value], data[data[:, feature_index] > feature_value]

                if len(left_data) != 0 and len(right_data) != 0:
                    left_data_preds, right_data_preds, preds = left_data[:, -1], right_data[:, -1], data[:, -1]
                    variance_reduction = np.std(preds) - (len(left_data_preds) / len(preds) * np.std(left_data_preds) + 
                                                                     len(right_data_preds) / len(preds) * np.std(right_data_preds))

                    if variance_reduction > max_variance_reduction:
                        max_variance_reduction = variance_reduction

                        best_split_params[0] = left_data
                        best_split_params[1] = right_data
                        best_split_params[2] = feature_index
                        best_split_params[3] = feature_value
                        best_split_params[4] = variance_reduction
              
        return best_split_params
    
    def insert_tree(self, data, tree_depth = 0): 
        preds = data[:,-1]
        samples_num = len(data)

        if samples_num >= self.min_samples_split and tree_depth <= self.max_depth:

            left_data, right_data, feature_index, threshold_value, information_gain = self.find_best_split(data)

            if len(left_data) >= self.min_samples_leaf and len(right_data) >= self.min_samples_leaf:
         
                if information_gain > 0:
                    
                    left_subtree = self.insert_tree(left_data, tree_depth+1)
                    
                    right_subtree = self.insert_tree(right_data, tree_depth+1)
                    
                    return Node(feature_index, threshold_value, 
                                left_subtree, right_subtree, information_gain)
            

        leaf_value = np.mean(preds)
    
        return Node(class_value = leaf_value)

    def fit(self, samples, preds):
        self.features_num = samples.shape[1]

        self.tree = self.insert_tree(data = np.concatenate((samples, np.array(preds, ndmin = 2).T), axis = 1))


    def predict(self, data):

        return [self.predict_sample(sample, self.tree) for sample in data]
    
    def predict_sample(self, sample, tree):
      
        if tree.class_value != None: return tree.class_value

        if  sample[tree.feature_index] <= tree.threshold_value:
            return self.predict_sample(sample, tree.left)
        else:
            return self.predict_sample(sample, tree.right)

    def print_tree(self, tree = None, tree_depth = 0):
        if tree == None: tree = self.tree

        if tree.threshold_value != None:
            
            self.print_tree(tree.left, tree_depth + 1)
            print(f"{' ' * 4 * tree_depth} -> {tree.threshold_value}")
            self.print_tree(tree.right, tree_depth + 1)

        elif tree.class_value != None:
            print(f"{' ' * 4 * tree_depth} -> {tree.class_value}")
            



if __name__ == "__main__":
    data = generate_regression_data(100)
    splited_data = split_data(data, ratio = 0.25)
    x_train, x_test, y_train, y_test = splited_data[0][:, :1], splited_data[1][:, :1], splited_data[0][:, 1], splited_data[1][:, 1]

    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    y_pred = dtr.predict(x_test)

    indices = np.argsort(x_test[:, 0])

    xs = np.array(x_test)[indices]
    ys = np.array(y_pred)[indices]
    
    f = plt.figure(figsize = (16 * 0.5, 9 * 0.5))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(x_test, y_test, 'o')
    ax.plot(xs, ys, 'r')
    ax.set_title('Decision Tree Regressor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
   
    plt.grid()
    plt.show()