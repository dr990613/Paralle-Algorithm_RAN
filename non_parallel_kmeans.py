import numpy as np


class KMeans:
    def __init__(self, K, max_iters=100, tol=1e-4, init_method="random"):
        """
        Initialize the KMeans algorithm.

        Parameters:
        - K: Number of clusters.
        - max_iters: Maximum number of iterations for convergence.
        - tol: Tolerance for convergence (i.e., minimum centroid shift).
        - init_method: Initialization method for centroids, either 'random' or 'kmeans++'.
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X: The dataset (2D array or DataFrame) to fit the model.

        Returns:
        - labels: The cluster labels for each data point.
        - centroids: The final centroids after convergence.
        - losses: A list of centroid shifts (for benchmarking or analysis).
        """
        self.centroids = self._initialize_centroids(X)
        losses = []  # List to store centroid shifts for each iteration

        for i in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._compute_centroids(X, labels)

            # Calculate the shift (loss) of centroids between iterations
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            losses.append(centroid_shift)

            if centroid_shift < self.tol:
                print(f"Converged in {i + 1} iterations.")
                break
            self.centroids = new_centroids

        self.labels = labels
        return self.labels, self.centroids, losses

    def _initialize_centroids(self, X):
        """
        Initialize centroids based on the chosen method ('random' or 'kmeans++').

        Parameters:
        - X: The dataset (2D array or DataFrame).

        Returns:
        - centroids: The initialized centroids.
        """
        if self.init_method == "random":
            random_indices = np.random.choice(len(X), self.K, replace=False)
            centroids = X[random_indices]
        elif self.init_method == "kmeans++":
            centroids = self._initialize_kmeans_plus_plus(X)
        return centroids

    def _initialize_kmeans_plus_plus(self, X):
        """
        Initialize centroids using the KMeans++ method.

        Parameters:
        - X: The dataset (2D array or DataFrame).

        Returns:
        - centroids: The initialized centroids.
        """
        centroids = [X[np.random.choice(len(X))]]
        for _ in range(1, self.K):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            next_centroid = X[np.random.choice(len(X), p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _assign_labels(self, X):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - X: The dataset (2D array or DataFrame).

        Returns:
        - labels: The cluster labels for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _compute_centroids(self, X, labels):
        """
        Compute the new centroids by averaging the data points in each cluster.

        Parameters:
        - X: The dataset (2D array or DataFrame).
        - labels: The cluster labels for each data point.

        Returns:
        - centroids: The new centroids.
        """
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.K)])
        return centroids
