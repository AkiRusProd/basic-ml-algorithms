import numpy as np
from utils import generate_clusterization_data, split_data
from metrics import accuracy



class NaiveBayesClassifier():

    def __init__(self):
        pass
        
        
    def normal_distribution(self, x, mean, var):
        
        return np.exp(-0.5 * np.power((x - mean) / var, 2)) / var * np.sqrt(2 * np.pi)


    def map_estimation(self, sample):
        classes_probabilities = [[] for _ in range(self.classes_num)]

        for i in range(self.classes_num):
            classes_probabilities[i] = np.sum(np.log(self.normal_distribution(sample, self.mean[i], self.var[i]))) + np.log(self.priors[i])

        return np.argmax(classes_probabilities)


    def fit(self, data, labels):
        self.samples_num, self.params_num = data.shape
        self.classes_num = len(np.unique(labels))

        self.mean = np.zeros((self.classes_num, self.params_num))
        self.var = np.zeros((self.classes_num, self.params_num))
        self.priors = np.zeros((self.classes_num))

        for i in range(self.classes_num):
            self.mean[i] = data[i == labels].mean(axis = 0)
            self.var[i] = data[i == labels].var(axis = 0)
            self.priors[i] = len(data[i == labels]) / self.samples_num


    def predict(self, data):
        return [self.map_estimation(sample) for sample in data]

   


if __name__ == "__main__":
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 300)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print(f"accuracy: {accuracy(y_test, y_pred) * 100}%")
