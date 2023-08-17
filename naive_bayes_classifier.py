import numpy as np
from utils import generate_clusterization_data, split_data, accuracy



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
    generated_data, generated_labels = generate_clusterization_data(n_clusters = 2)
    train_data, test_data, train_labels, test_labels =  split_data(generated_data, generated_labels, ratio = 0.25)

    nb = NaiveBayesClassifier()
    nb.fit(train_data, train_labels)
    predicted_labels = nb.predict(test_data)

    print(f"accuracy: {accuracy(test_labels, predicted_labels) * 100}%")
