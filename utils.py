import numpy as np

def generate_clusterization_data(n_clusters, n_samples = None):
    data = np.array([], ndmin = 2)
    labels = np.array([], ndmin = 2)

    for i in range(n_clusters):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        n_samples = np.random.randint(40, 100, 1) if n_samples is None else n_samples

        generated_cluster = np.random.multivariate_normal(mean, cov, n_samples)
        data = np.concatenate([data, generated_cluster]) if data.size else generated_cluster

        generated_claster_labels =np.full(n_samples, i, dtype=int)
        labels = np.concatenate([labels, generated_claster_labels]) if labels.size else generated_claster_labels

    return data, labels

# def generate_regression_data(n_samples = 30, beta = 10, variance_reduction = 10):

#     e = (np.random.randn(n_samples) * variance_reduction).round(decimals = 1)

#     x = (np.random.rand(n_samples) * n_samples)
#     y = (np.random.rand(n_samples) * n_samples)

#     z = x * beta + y * beta + e
#     x, y, z = np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1), np.expand_dims(z, axis = 1)

#     return np.concatenate((x, y, z), axis=1)

def generate_regression_data(n_samples=30, beta=10, variance_reduction=10, n_clusters=5):
    x = np.arange(0, n_samples * n_clusters)[..., None]
    y = np.array([])

    start = 0; end = 100; max_diff = 10

    for i in range(n_clusters):
        min_val = np.random.randint(start, end - max_diff + 1)
        max_val = min_val + np.random.randint(1, max_diff + 1)

        y_cluster = np.random.uniform(min_val, max_val, n_samples)
        y = np.concatenate((y, y_cluster))

    y = y[:, None]

    return np.concatenate((x, y), axis=1)



def split_data(data, labels = None, ratio = 0.25):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data = data[indices].reshape(data.shape)
    train_data, test_data = data[:int(len(data) * (1 - ratio))], data[-int(len(data) * ratio):]

    if labels is not None:
        labels = labels[indices].reshape(labels.shape)
        train_labels, test_labels = labels[:int(len(data) * (1 - ratio))], labels[-int(len(data) * ratio):]
    
        return train_data, test_data, train_labels, test_labels
    else:
        return train_data, test_data

def accuracy(targets, predictions):
    return np.equal(targets, predictions).mean()
