import numpy as np
import multiprocessing as mp


class ParallelKMeans:
    def __init__(self, K, max_iters=100, tol=1e-4, init_method="random", num_processes=None):
        """
        Initialize the ParallelKMeans algorithm.

        Parameters:
        - K: Number of clusters.
        - max_iters: Maximum number of iterations for convergence.
        - tol: Tolerance for convergence (i.e., minimum centroid shift).
        - init_method: Initialization method for centroids, either 'random' or 'kmeans++'.
        - num_processes: Number of processes for parallelization. If None, use all available cores.
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.num_processes = num_processes or mp.cpu_count()
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit the ParallelKMeans model to the data.

        Parameters:
        - X: The dataset (2D array or DataFrame) to fit the model.

        Returns:
        - labels: The cluster labels for each data point.
        - centroids: The final centroids after convergence.
        - losses: List of centroid shifts for each iteration.
        """
        self.centroids = self._initialize_centroids(X)
        losses = []

        for i in range(self.max_iters):
            # Parallel label assignment
            labels = self._parallel_assign_labels(X)
            # Parallel centroid update
            new_centroids = self._parallel_update_centroids(X, labels)

            # Calculate centroid shift as a measure of convergence
            centroid_shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            losses.append(centroid_shift)

            if centroid_shift < self.tol:
                print(f"Converged in {i + 1} iterations.")
                break
            self.centroids = new_centroids

        self.labels = labels
        return self.labels, self.centroids, losses

    def _initialize_centroids(self, X):
        """
        Initialize centroids using random or k-means++ initialization.
        """
        if self.init_method == "random":
            random_indices = np.random.choice(len(X), self.K, replace=False)
            centroids = X[random_indices]
        elif self.init_method == "kmeans++":
            centroids = self._kmeans_plus_plus(X)
        return centroids

    def _kmeans_plus_plus(self, X):
        """
        Initialize centroids using the k-means++ method.
        """
        centroids = []
        centroids.append(X[np.random.randint(len(X))])
        for _ in range(1, self.K):
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            prob_dist = dist_sq / dist_sq.sum()
            next_centroid = X[np.random.choice(len(X), p=prob_dist)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _parallel_assign_labels(self, X):
        """
        Assign labels to the data points in parallel.

        Parameters:
        - X: The dataset.

        Returns:
        - labels: Array of labels assigned to each data point.
        """
        # Split data into chunks for parallel processing
        num_chunks = self.num_processes
        chunks = np.array_split(X, num_chunks)

        # Parallelize label assignment
        with mp.Pool(self.num_processes) as pool:
            results = pool.map(self._assign_labels_chunk, chunks)

        # Combine results from all chunks
        labels = np.concatenate(results, axis=0)
        return labels

    def _assign_labels_chunk(self, chunk):
        """
        Assign labels for a chunk of data.
        """
        distances = np.linalg.norm(chunk[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _parallel_update_centroids(self, X, labels):
        """
        Update centroids in parallel.

        Parameters:
        - X: The dataset.
        - labels: The labels assigned to the data points.

        Returns:
        - centroids: The new centroids.
        """
        # Split data into chunks for parallel processing
        num_chunks = self.num_processes
        chunks = np.array_split(X, num_chunks)
        label_chunks = np.array_split(labels, num_chunks)

        # Parallelize centroid update
        with mp.Pool(self.num_processes) as pool:
            results = pool.starmap(self._update_centroids_chunk, zip(chunks, label_chunks))

        # Combine results from all chunks
        new_centroids = np.mean(results, axis=0)
        return new_centroids

    def _update_centroids_chunk(self, chunk, labels_chunk):
        """
        Update centroids for a chunk of data.
        """
        centroids = np.zeros((self.K, chunk.shape[1]))
        for k in range(self.K):
            points_in_cluster = chunk[labels_chunk == k]
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)
        return centroids
