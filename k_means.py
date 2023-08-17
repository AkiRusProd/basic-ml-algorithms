import numpy as np
import matplotlib.pyplot as plt
from utils import generate_clusterization_data



"""K-means"""

class KMeans():

    def __init__(self, k):
        self.k = k
        self.centroids = []
        self.data = None
        
        
    def euclidean_distance(self, vector1, vector2):

        return np.linalg.norm(vector1 - vector2)


    def initialize_centroids(self):
        # k-means++ centroids initialization
        self.centroids.append(self.data[np.random.randint(len(self.data))])

        for _ in range(self.k - 1):
            min_distances = []

            for sample in self.data:
                min_distance = np.asfarray([self.euclidean_distance(sample, centroid) for centroid in self.centroids]).min()

                min_distances.append(min_distance)

            self.centroids.append(self.data[np.argmax(min_distances)])


    def fit(self, data):
        self.data = data

        self.initialize_centroids()
        previous_centroids = None
        
        while np.not_equal(self.centroids, previous_centroids).any():
            previous_centroids = self.centroids.copy()

            clusters_per_centroids = [[] for _ in range(self.k)]
          
            for sample in self.data:
                distances = np.asfarray([self.euclidean_distance(sample, centroid) for centroid in self.centroids])

                clusters_per_centroids[np.argmin(distances)].append(sample)

            self.centroids = [np.mean(cluster, axis=0) for cluster in clusters_per_centroids]
            
        return np.asfarray(self.centroids)

    def predict(self, sample):
        distances = np.asfarray([self.euclidean_distance(sample, centroid) for centroid in self.centroids])

        return np.argmin(distances)

        

if __name__ == '__main__':
    X_train, y_train = generate_clusterization_data(n_clusters = 3)

    k_means = KMeans(k = 3)

    centroids = k_means.fit(X_train)

    plt.scatter(X_train[:,0], X_train[:,1], s = 40, c = y_train,  cmap = plt.cm.spring, edgecolors = 'k')
    plt.scatter(centroids[:,0], centroids[:,1], s = 200, color = 'red' , marker = '*', edgecolors = 'k', label = 'centroids')

    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()

