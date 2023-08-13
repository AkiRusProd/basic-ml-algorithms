import numpy as np 



def generate_data(clusters_num, samples_num):
    data = np.array([], ndmin = 2)
    labels = np.array([], ndmin = 2)

    for i in range(clusters_num):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        samples_num = np.random.randint(40, 100, 1) if samples_num is None else samples_num

        generated_cluster = np.random.multivariate_normal(mean, cov, samples_num)
        data = np.concatenate([data, generated_cluster]) if data.size else generated_cluster

        generated_claster_labels =np.full(samples_num, i, dtype=int)
        labels = np.concatenate([labels, generated_claster_labels]) if labels.size else generated_claster_labels

    return data, labels




def split_data(data, labels, ratio):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data, labels = data[indices].reshape(data.shape), labels[indices].reshape(labels.shape)

    train_data, test_data = data[:int(len(data) * (1 - ratio))], data[-int(len(data) * ratio):]
    train_labels, test_labels = labels[:int(len(data) * (1 - ratio))], labels[-int(len(data) * ratio):]

    return train_data, test_data, train_labels, test_labels






"""CART Decision Tree"""
#https://en.wikipedia.org/wiki/Decision_tree_learning

class Node():
    def __init__(self, feature_index=None, threshold_value = None, left = None, right = None, information_gain = None, class_value = None):

        self.feature_index = feature_index
        self.threshold_value = threshold_value
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.class_value = class_value


class DecisionTreeClassifier():

    def __init__(self, min_samples_split = 2, min_samples_leaf = 2, max_depth = 2, criterion = 'gini'):
        self.tree = None
      
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion

    def compute_gini_index(self, labels):
        sigma = 0
        for class_value in np.unique(labels):
            class_probability = len(labels[labels == class_value]) / len(labels)
            sigma += class_probability**2
        return 1 - sigma


    def compute_entropy(self, labels): 
        entropy = 0
        for class_value in np.unique(labels):
            class_probability = len(labels[labels == class_value]) / len(labels)
            entropy += - class_probability * np.log2(class_probability)
        return entropy


    def find_best_split(self, data):
        max_information_gain = -np.inf
        
        best_split_params = [[] for _ in range(5)]

        for feature_index in range(self.features_num):
            for feature_value in np.unique(data[:, feature_index]):
                left_data, right_data = data[data[:, feature_index] <= feature_value], data[data[:, feature_index] > feature_value]

                if len(left_data) != 0 and len(right_data) != 0:
                    left_data_labels, right_data_labels, labels = left_data[:, -1], right_data[:, -1], data[:, -1]
                    information_gain = self.criterion_func(labels) - (len(left_data_labels) / len(labels) * self.criterion_func(left_data_labels) + 
                                                                     len(right_data_labels) / len(labels) * self.criterion_func(right_data_labels))

                    if information_gain > max_information_gain:
                        max_information_gain = information_gain

                        best_split_params[0] = left_data
                        best_split_params[1] = right_data
                        best_split_params[2] = feature_index
                        best_split_params[3] = feature_value
                        best_split_params[4] = information_gain
              
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
       
        if self.criterion == 'gini':
            self.criterion_func = self.compute_gini_index
        elif self.criterion == 'entropy':
            self.criterion_func = self.compute_entropy
        else:
            raise SystemExit(f'Criterion with name "{self.criterion}" not found') 

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
            



def acc(targets, predictions):

    return np.equal(targets, predictions).mean()




if __name__ == "__main__":
    generated_data, generated_labels = generate_data(clusters_num = 2, samples_num = 300) #clusters equal classes
    x_train, x_test, y_train, y_test =  split_data(generated_data, generated_labels, ratio = 0.25)

    dsc = DecisionTreeClassifier()

    dsc.fit(x_train, y_train)
    y_pred = dsc.predict(x_test)

    dsc.print_tree()


    print(f"accuracy: {acc(y_test, y_pred) * 100}%")
