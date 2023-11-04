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

    def __init__(self, min_samples_split = 2, min_samples_leaf = 2, max_depth = 5, criterion = 'variance reduction'):
        self.tree = None
      
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion
       
    def compute_variance_reduction(self, left_data_preds, right_data_preds, data_preds):
        return np.std(data_preds) - (len(left_data_preds) / len(data_preds) * np.std(left_data_preds) +
                                     len(right_data_preds) / len(data_preds) * np.std(right_data_preds))

    def compute_mse(self, left_data_preds, right_data_preds, data_preds):
        left_part = np.sum((left_data_preds - np.mean(left_data_preds)) ** 2) if len(left_data_preds) > 0 else 0
        right_part = np.sum((right_data_preds - np.mean(right_data_preds)) ** 2) if len(right_data_preds) > 0 else 0
        mse = 1/len(data_preds) * (left_part + right_part)
        return mse

    def compute_mae(self, left_data_preds, right_data_preds, data_preds):
        left_part = np.sum(np.abs(left_data_preds - np.mean(left_data_preds))) if len(left_data_preds) > 0 else 0
        right_part = np.sum(np.abs(right_data_preds - np.mean(right_data_preds))) if len(right_data_preds) > 0 else 0
        mae = 1/len(data_preds) * (left_part + right_part)
        return mae

    # def compute_sse(self, left_data_preds, right_data_preds, _):
    #     #NOTE: Almost the same is mse (https://stats.stackexchange.com/questions/220350/regression-trees-how-are-splits-decided)
    #     #I don't want to add it, because it doesn't make sense.
    #     left_part = np.sum((left_data_preds - np.mean(left_data_preds)) ** 2) if len(left_data_preds) > 0 else 0
    #     right_part = np.sum((right_data_preds - np.mean(right_data_preds)) ** 2) if len(right_data_preds) > 0 else 0
    #     sse = left_part + right_part
    #     return sse

    def find_best_split(self, data):
        best_score = np.inf if self.criterion != 'variance reduction'  else -np.inf
        
        best_split_params = [[] for _ in range(5)]

        for feature_index in range(self.features_num):
            for feature_value in np.unique(data[:, feature_index]):
                left_data, right_data = data[data[:, feature_index] <= feature_value], data[data[:, feature_index] > feature_value]

                if len(left_data) != 0 and len(right_data) != 0:
                    left_data_preds, right_data_preds, preds = left_data[:, -1], right_data[:, -1], data[:, -1]
                    score = self.criterion_func(left_data_preds, right_data_preds, preds)

                    # NOTE: we are minimizing criterion if it's mse or mae and maximizing if it's variance reduction
                    if (self.criterion != 'variance reduction' and score < best_score) or (self.criterion == 'variance reduction' and score > best_score):
                        best_score = score
                        best_split_params[0] = left_data
                        best_split_params[1] = right_data
                        best_split_params[2] = feature_index
                        best_split_params[3] = feature_value
                        best_split_params[4] = score
              
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

        if self.criterion == 'mse':
            self.criterion_func = self.compute_mse
        elif self.criterion == 'mae':
            self.criterion_func = self.compute_mae
        # elif self.criterion == 'sse':
        #     self.criterion_func = self.compute_sse
        elif self.criterion == 'variance reduction':
            self.criterion_func = self.compute_variance_reduction
        else:
            raise SystemExit(f'Criterion with name "{self.criterion}" not found') 

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
    X_train, y_train = generate_regression_data(100)
    X_train, X_test, y_train, y_test = split_data(X_train, y_train, ratio = 0.25)

    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train[:, 0])
    y_pred = dtr.predict(X_test)

    indices = np.argsort(X_test[:, 0])

    xs = np.array(X_test)[indices]
    ys = np.array(y_pred)[indices]
    
    f = plt.figure(figsize = (16 * 0.5, 9 * 0.5))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(X_test, y_test, 'o')
    ax.plot(xs, ys, 'r')
    ax.set_title('Decision Tree Regressor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
   
    plt.grid()
    plt.show()