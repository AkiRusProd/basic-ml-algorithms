import numpy as np 
import matplotlib.pyplot as plt
 

def generate_dataset(n = 30, beta = 10, variance_reduction = 10):

    e = (np.random.randn(n) * variance_reduction).round(decimals = 1)

    x = (np.random.rand(n) * n)
    y = (np.random.rand(n) * n)

    z = x * beta + y * beta + e
    x, y, z = np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1), np.expand_dims(z, axis = 1)

    return np.concatenate((x, y, z), axis=1)
 
data = generate_dataset(200)



def split_data(data, ratio):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data = data[indices].reshape(data.shape)

    train_data, test_data = data[:int(len(data) * (1 - ratio))], data[-int(len(data) * ratio):]
   
    return train_data[:, :2], test_data[:, :2], train_data[:, 2], test_data[:, 2]


x_train, x_test, y_train, y_test = split_data(data, ratio = 0.25)



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
                    left_data_labels, right_data_labels, labels = left_data[:, -1], right_data[:, -1], data[:, -1]
                    variance_reduction = np.std(labels) - (len(left_data_labels) / len(labels) * np.std(left_data_labels) + 
                                                                     len(right_data_labels) / len(labels) * np.std(right_data_labels))

                    if variance_reduction > max_variance_reduction:
                        max_variance_reduction = variance_reduction

                        best_split_params[0] = left_data
                        best_split_params[1] = right_data
                        best_split_params[2] = feature_index
                        best_split_params[3] = feature_value
                        best_split_params[4] = variance_reduction
              
        return best_split_params
    
    def insert_tree(self, data, tree_depth = 0): 
        labels = data[:,-1]
        samples_num = len(data)

        if samples_num >= self.min_samples_split and tree_depth <= self.max_depth:

            left_data, right_data, feature_index, threshold_value, information_gain = self.find_best_split(data)

            if len(left_data) >= self.min_samples_leaf and len(right_data) >= self.min_samples_leaf:
         
                if information_gain > 0:
                    
                    left_subtree = self.insert_tree(left_data, tree_depth+1)
                    
                    right_subtree = self.insert_tree(right_data, tree_depth+1)
                    
                    return Node(feature_index, threshold_value, 
                                left_subtree, right_subtree, information_gain)
            

        leaf_value = np.argmax(np.bincount(labels.astype(int)))
    
        return Node(class_value = leaf_value)

    def fit(self, samples, labels):
        self.features_num = samples.shape[1]

        self.tree = self.insert_tree(data = np.concatenate((samples, np.array(labels, ndmin = 2).T), axis = 1))


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
            




dsr = DecisionTreeRegressor()

dsr.fit(x_train, y_train)
y_predicted = dsr.predict(x_test)
dsr.print_tree()



fig = plt.figure()
ax = fig.gca(projection ='3d')
 
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, 
            label ='train values', s = 5, color ="dodgerblue")

ax.scatter(x_test[:, 0], x_test[:, 1], y_test,
                label ='test values', s = 5, color ="blue")
 
ax.scatter(x_test[:, 0], x_test[:, 1], y_predicted,
                label ='predicted values', s = 5, color ="orange")
ax.legend()
ax.view_init(45, 0)
 
plt.show()
