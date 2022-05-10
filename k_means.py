import numpy as np
import matplotlib.pyplot as plt


def generate_data(clusters_num):
    data = np.array([])
    labels = np.array([])

    for i in range(clusters_num):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        samples_num = np.random.randint(30, 100, 1)

        generated_cluster = np.random.multivariate_normal(mean, cov, samples_num)
        data = np.concatenate([data, generated_cluster]) if data.size else generated_cluster

        generated_claster_labels =np.full(samples_num, i, dtype=int)
        labels = np.concatenate([labels, generated_claster_labels]) if labels.size else generated_claster_labels

    return data, labels

data, labels = generate_data(clusters_num = 3)


"""K-means"""

class K_Means():

    def __init__(self, k):
        self.k = k
        self.centroids = []
        self.data = None
        
        
    def euclidean_distance(self, vector1, vector2):

        return np.linalg.norm(vector1 - vector2)


    def initialize_centroids(self):

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

        


k_means = K_Means(k = 3)

centroids = k_means.fit(data)

plt.scatter(data[:,0], data[:,1], s = 40, c = labels,  cmap = plt.cm.spring, edgecolors = 'k')
plt.scatter(centroids[:,0], centroids[:,1], s = 200, color = 'red' , marker = '*', edgecolors = 'k', label = 'centroids')


plt.legend(loc=2)
plt.grid(True, linestyle='-', color='0.75')
plt.show()

