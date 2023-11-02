import numpy as np
import matplotlib.pyplot as plt
from utils import generate_clusterization_data




"""K-nearest neighbors"""
class KNNClassifier():
    def __init__(self, n_neighbors=3, metric='euclidean', weights = 'uniform'):
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights
        self.distance = {
            'euclidean': lambda x, y: np.linalg.norm(x - y),
            'manhattan': lambda x, y: np.sum(np.abs(x - y)),
            'chebyshev': lambda x, y: np.max(np.abs(x - y)),
            'cosine': lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
            'canberra': lambda x, y: np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-15)),
            'braycurtis': lambda x, y: np.sum(np.abs(x - y) / (np.sum(np.abs(x)) + np.sum(np.abs(y)) + 1e-15)),
            'hamming': lambda x, y: np.average(np.atleast_1d(x) != np.atleast_1d(y))

        }[metric]
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.distance(x, y) for y in self.X])
            k_nearest_neighbors = self.y[distances.argsort()[:self.k]]

            if self.weights == 'distance':
                k_weights = np.array([1 / (distance + 1E-15) for distance in np.sort(distances)[:self.k]]) #distances[distances.argsort()[:self.k]]
                k_weights = k_weights / np.sum(k_weights)
             
                predictions.append(np.argmax(np.bincount(k_nearest_neighbors, weights = k_weights)))
            elif self.weights == 'uniform':
                predictions.append(np.argmax(np.bincount(k_nearest_neighbors)))

        return np.array(predictions)

       

if __name__ == "__main__":
    X_train, y_test = generate_clusterization_data(n_clusters = 3, n_samples = 30)

    knn = KNNClassifier(n_neighbors = 5)
    knn.fit(X_train, y_test)


    x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max() + 1
    y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max() + 1

    x_grid, y_grid = np.meshgrid(np.arange(x_min,x_max,.1),np.arange(y_min,y_max,.1))

    predictions = knn.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

    predictions = predictions.reshape(x_grid.shape)

    plt.pcolormesh(x_grid, y_grid, predictions, cmap = plt.cm.Pastel2)
    plt.scatter(X_train[:,0], X_train[:,1], s = 80, c = y_test,  cmap = plt.cm.spring, edgecolors = 'k')
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())

    plt.title("KNN Classifier")
    
    plt.show()