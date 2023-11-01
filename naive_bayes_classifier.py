import numpy as np
from utils import generate_clusterization_data, split_data
from metrics import accuracy



class NaiveBayesClassifier():
        
    def norm_pdf(self, x, mean, var):
        return np.exp(-0.5 * np.power((x - mean) / np.sqrt(var), 2)) / (np.sqrt(2 * np.pi) * np.sqrt(var))


    def map_estimation(self, x):
        posteriors = [] # classes probabilities

        for i, c in enumerate(self.classes):
            # Logarithms are used to prevent precision issues when dealing with very small probabilities 
            # and to speed up computations by transforming the multiplication of probabilities into the sum of their logarithms.

            posteriors.append(np.sum(np.log(self.norm_pdf(x, self.mean[i], self.var[i]))) + np.log(self.priors[i])) # class_conditional + prior

        return self.classes[np.argmax(posteriors)]


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros((n_classes))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i] = X_c.mean(axis=0)
            self.var[i] = X_c.var(axis=0)
            self.priors[i] = len(X_c) / n_samples


    def predict(self, X):
        return [self.map_estimation(x) for x in X]


if __name__ == "__main__":
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 300)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print(f"accuracy: {accuracy(y_test, y_pred) * 100}%")
