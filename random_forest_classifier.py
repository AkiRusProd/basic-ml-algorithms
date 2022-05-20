import numpy as np
from decision_tree_classifier import DecisionTreeClassifier


def generate_data(clusters_num):
    data = np.array([], ndmin = 2)
    labels = np.array([], ndmin = 2)

    for i in range(clusters_num):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        samples_num = np.random.randint(40, 100, 1)

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



def acc(targets, predictions):

    return np.equal(targets, predictions).mean()



if __name__ == "__main__":
    generated_data, generated_labels = generate_data(clusters_num = 2) #clusters equal classes
    train_data, test_data, train_labels, test_labels =  split_data(generated_data, generated_labels, ratio = 0.25)

    rfc = RandomForestClassifier()

    rfc.fit(train_data, train_labels)
    predicted_labels = rfc.predict(test_data)


    print(f"accuracy: {acc(test_labels, predicted_labels) * 100}%")
