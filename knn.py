import numpy as np
import matplotlib.pyplot as plt
from utils import generate_clusterization_data




"""K-nearest neighbors"""

class KNN_Classifier():

    def __init__(self, k = 5) -> None:
        self.k = k
        self.data = None
        self.labels = None


    def euclidean_distance(self, vector1, vector2):

        return np.linalg.norm(vector1 - vector2)


    def find_nearest_neighbour(self, this_sample):
        distances = np.asfarray([self.euclidean_distance(this_sample, sample) for sample in self.data])

        indexes = distances.argsort()

        neighbours = self.labels[indexes]

        k_neighbours = neighbours[:self.k]
        
        return np.argmax(np.bincount(k_neighbours.astype(int)))

    def fit(self, data, labels) -> None:
        self.data = data
        self.labels = labels

    def predict(self, new_data):

        return np.asfarray([self.find_nearest_neighbour(sample) for sample in new_data])

       

if __name__ == "__main__":
    data, labels = generate_clusterization_data(n_clusters = 3, n_samples = 30)

    knn = KNN_Classifier(k = 5)
    knn.fit(data, labels)


    x_min, x_max = data[:,0].min()-1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min()-1, data[:,1].max() + 1

    x_grid, y_grid = np.meshgrid(np.arange(x_min,x_max,.1),np.arange(y_min,y_max,.1))

    predictions = knn.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

    predictions = predictions.reshape(x_grid.shape)

    plt.pcolormesh(x_grid, y_grid, predictions, cmap = plt.cm.Pastel2)
    plt.scatter(data[:,0], data[:,1], s = 80, c = labels,  cmap = plt.cm.spring, edgecolors = 'k')
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())

    plt.title("KNN Classifier")
    
    plt.show()