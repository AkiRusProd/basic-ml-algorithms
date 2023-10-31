import numpy as np

def generate_clusterization_data(n_clusters, n_samples = None):
    X = np.array([], ndmin = 2)
    y = np.array([], ndmin = 2)

    for i in range(n_clusters):
        mean = np.random.randint(-10, 10, 2)
        cov =  np.random.randint(-10, 10, [2, 2])
        n_samples = np.random.randint(40, 100, 1) if n_samples is None else n_samples

        X_new = np.random.multivariate_normal(mean, cov, n_samples)
        X = np.concatenate([X, X_new]) if X.size else X_new

        y_new = np.full(n_samples, i, dtype=int)
        y = np.concatenate([y, y_new]) if y.size else y_new

    return X, y

# def generate_regression_data(n_samples = 30, beta = 10, variance_reduction = 10):

#     e = (np.random.randn(n_samples) * variance_reduction).round(decimals = 1)

#     x = (np.random.rand(n_samples) * n_samples)
#     y = (np.random.rand(n_samples) * n_samples)

#     z = x * beta + y * beta + e
#     x, y, z = np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1), np.expand_dims(z, axis = 1)

#     return np.concatenate((x, y, z), axis=1)

def generate_regression_data(n_samples=30, n_clusters=5):
    X = np.arange(0, n_samples * n_clusters)[..., None]
    y = np.array([])

    start = 0; end = 100; max_diff = 10

    for i in range(n_clusters):
        min_val = np.random.randint(start, end - max_diff + 1)
        max_val = min_val + np.random.randint(1, max_diff + 1)

        y_cluster = np.random.uniform(min_val, max_val, n_samples)
        y = np.concatenate((y, y_cluster))

    y = y[:, None]

    return X, y

def generate_linear_regression_data(num_samples = 100, num_features = 1, noise=0.1):
    # Generate random feature matrix X
    X = np.random.rand(num_samples, num_features)

    # Generate true coefficients for the linear regression model
    true_coefficients = np.random.rand(num_features, 1)

    # Generate noise term
    noise = noise * np.random.randn(num_samples, 1)

    # Generate target variable y
    y = np.dot(X, true_coefficients) + noise

    return X, y, true_coefficients




def split_data(X, y = None, ratio = 0.25):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices].reshape(X.shape)
    X_train, X_test = X[:int(len(X) * (1 - ratio))], X[-int(len(X) * ratio):]

    y = y[indices].reshape(y.shape)
    y_train, y_test = y[:int(len(X) * (1 - ratio))], y[-int(len(X) * ratio):]

    return X_train, X_test, y_train, y_test


