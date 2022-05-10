import numpy as np


def generate_data(clusters_num):
    data = np.array([])
    labels = np.array([])

    for i in range(clusters_num):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        samples_num = np.random.randint(40, 100, 1)

        generated_cluster = np.random.multivariate_normal(mean, cov, samples_num)
        data = np.concatenate([data, generated_cluster]) if data.size else generated_cluster

        generated_claster_labels =np.full(samples_num, i, dtype=int)
        labels = np.concatenate([labels, generated_claster_labels]) if labels.size else generated_claster_labels

    return data, labels

generated_data, generated_labels = generate_data(clusters_num = 2) #cluster equal class


def split_data(data, labels, ratio):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data, labels = data[indices].reshape(data.shape), labels[indices].reshape(labels.shape)

    train_data, test_data = data[:int(len(data) * (1 - ratio))], data[-int(len(data) * ratio):]
    train_labels, test_labels = labels[:int(len(data) * (1 - ratio))], labels[-int(len(data) * ratio):]

    return train_data, test_data, train_labels, test_labels



train_data, test_data, train_labels, test_labels =  split_data(generated_data, generated_labels, ratio = 0.25)



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

   


def acc(targets, predictions):
    return np.equal(targets, predictions).mean()


nb = NaiveBayesClassifier()
nb.fit(train_data, train_labels)
predicted_labels = nb.predict(test_data)

print(f"accuracy: {acc(test_labels, predicted_labels) * 100}%")
