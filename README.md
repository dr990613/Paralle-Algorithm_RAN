# Paralle-Algorithm_RAN

## Algorithm and Parallelization Method

The K-Means algorithm is a classic clustering algorithm that aims to partition a set of points into K clusters, where each point belongs to the cluster with the nearest mean.

- **Non-Parallel Version**: A standard implementation of K-Means where each step of the algorithm (assigning points to centroids and updating centroids) is executed sequentially.
- **Parallel Version**: The parallelization is applied to the two key steps:
  - **Assigning points to centroids**: This step is parallelized by distributing the task of calculating distances between points and centroids across multiple threads or processes.
  - **Updating centroids**: The calculation of the new centroids for each cluster is parallelized by splitting the work of summing up the points in each cluster.

We use **Python's `multiprocessing`** and **`joblib`** to implement the parallelization.

## Instructions to Reproduce Results

### Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `seaborn`
- `tqdm`

### Data Preparation

1. Download the Titanic dataset from [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data).
2. Place the dataset file (`train.csv`) in the `data/` directory.

### Running the Code

1. **Non-Parallel K-Means**:
    - Run the following Python script to execute the non-parallel version of K-Means:

```Bash
python non_parallel_kmeans.py
```
2. **Parallel K-Means**:
    - Run the parallelized version:

```Bash
python parallel_kmeans.py
```
3. **Cluster Analysis**:
    - After running the code, you can visualize the clusters using:

```Bash
python visualize_clusters.py
```

### Example Usage:

```Bash
python non_parallel_kmeans.py
python parallel_kmeans.py
python visualize_clusters.py
```

## Explanation of Parallelization

In this project, the K-Means algorithm is parallelized in two major steps:

1. **Assigning points to centroids**: The distance computation for each point to all centroids is done in parallel. Each point can be independently assigned to the closest centroid, so we parallelize this task across multiple threads/processes.
2. **Updating centroids**: The process of calculating the new centroids for each cluster is also parallelized. The new centroid for each cluster is computed by averaging the points assigned to that cluster, and this operation is performed concurrently for each cluster.

We used **Python's ****`multiprocessing`** and **`joblib`** libraries to parallelize the algorithm.

## Speedup Calculation

The speedup is defined as the ratio of the execution time of the non-parallel version to the execution time of the parallel version:

Speedup = Time x Non-Parallel/ Time (Parallel)


A plot showing the speedup as a function of the number of threads/processes will be generated during the experiments. The speedup calculation is used to evaluate the effectiveness of parallelization.

### Speedup Graph

A graph will be generated that shows the speedup achieved by using different numbers of threads or processes.
