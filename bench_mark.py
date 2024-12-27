import time
import numpy as np
import matplotlib.pyplot as plt
from non_parallel_kmeans import KMeans  # 导入非并行版的 KMeans
from parallel_kmeans import ParallelKMeans  # 导入并行版的 ParallelKMeans
from data_preprocessing import load_and_preprocess_data  # 导入数据处理函数


def run_non_parallel_kmeans(X, K, max_iters, tol):
    """
    运行非并行 KMeans 并记录时间。

    参数:
    - X: 数据集（NumPy 数组）
    - K: 聚类数量
    - max_iters: 最大迭代次数
    - tol: 收敛容忍度

    返回:
    - 运行时间（秒）
    """
    kmeans = KMeans(K=K, max_iters=max_iters, tol=tol)
    start_time = time.time()
    kmeans.fit(X)
    end_time = time.time()
    return end_time - start_time


def run_parallel_kmeans(X, K, max_iters, tol, num_processes):
    """
Run parallel KMeans and record the time.

Parameters:
- X: dataset (NumPy array)
- K: number of clusters
- max_iters: maximum number of iterations
- tol: convergence tolerance
- num_processes: number of parallel processes

Returns:
- Run time (seconds)
    """
    parallel_kmeans = ParallelKMeans(K=K, max_iters=max_iters, tol=tol, num_processes=num_processes)
    start_time = time.time()
    parallel_kmeans.fit(X)
    end_time = time.time()
    return end_time - start_time


def main():

    file_path = 'data.csv'  # 请根据需要调整文件路径
    data = load_and_preprocess_data(file_path)


    X = data.values


    K = 3
    max_iters = 100
    tol = 1e-4


    non_parallel_time = run_non_parallel_kmeans(X, K, max_iters, tol)


    num_processes_list = [1, 2, 4, 8, 16]
    parallel_times = []

    for num_processes in num_processes_list:
        parallel_time = run_parallel_kmeans(X, K, max_iters, tol, num_processes)
        parallel_times.append(parallel_time)


    speedup = [non_parallel_time / pt for pt in parallel_times]


    print(f"Non-parallel time: {non_parallel_time:.4f} seconds")
    for i, num_processes in enumerate(num_processes_list):
        print(f"Parallel with {num_processes} processes: {parallel_times[i]:.4f} seconds")
        print(f"Speedup with {num_processes} processes: {speedup[i]:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(num_processes_list, speedup, marker='o', color='b', linestyle='-', label='Speedup')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup of Parallel KMeans vs. Non-Parallel KMeans')
    plt.grid(True)
    plt.legend()
    plt.xticks(num_processes_list)
    plt.show()


if __name__ == "__main__":
    main()
