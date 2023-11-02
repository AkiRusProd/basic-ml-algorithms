import numpy as np
from utils import generate_clusterization_data, split_data
from metrics import accuracy


# https://en.wikipedia.org/wiki/Naive_Bayes_classifier

# Bayes theorem: P(c|x) = P(c) * P(x|c) / P(x) or posterior = prior * likelihood / evidence
# We ignore P(x) in the Bayes formula, as it doesn't affect which class has the highest probability (since the P(x) does not depend on C and property values X are given). 
# So we set P(x) = 1. (constant)): P(c|x) = P(c) * P(x|c) or posterior = prior * likelihood
class NaiveBayesClassifier():
        
    def norm_pdf(self, x, mean, var):
        return np.exp(-0.5 * np.power((x - mean) / np.sqrt(var), 2)) / (np.sqrt(2 * np.pi) * np.sqrt(var))


    def map_estimation(self, x):
        posteriors = [] # classes probabilities

        for i, c in enumerate(self.classes):
            # Logarithms are used to prevent precision issues when dealing with very small probabilities 
            # and to speed up computations by transforming the multiplication of probabilities into the sum of their logarithms.

            # P(c|x_1, x_2, ..., x_n) = P(c) * ∏(P(x_i|c)) =>
            # log(P(c|x1, x2, ..., xn)) = log(P(c)) + ∑(log(P(x_i|c))) =>
            # posterior = prior + class_conditional
           
            posteriors.append(np.log(self.priors[i]) + np.sum(np.log(self.norm_pdf(x, self.mean[i], self.var[i])))) 

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
