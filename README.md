### README

#### **Project Title**
K-means Clustering with Parallel and Non-Parallel Implementations

#### **Dataset Overview**
The dataset used in this project represents a comprehensive collection of used vehicle listings across the United States on Craigslist. It was initially built as a school project by scraping the Craigslist platform and later expanded to create a robust dataset. This dataset provides an extensive set of features, including information such as the vehicle's year, odometer reading, price, and geographical details (latitude and longitude).
https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

#### **Project Description**
This project demonstrates the implementation of the K-means clustering algorithm using both non-parallel and parallel approaches. The primary objective is to cluster the vehicle data into meaningful groups based on features such as year and odometer readings, while also showcasing the performance difference between the non-parallel and parallel implementations.

#### **Goals**
1. **Data Preprocessing**: Clean and preprocess the dataset to retain relevant numerical features, remove outliers, and standardize the data for clustering.
2. **K-means Clustering**:
   - Implement a non-parallel K-means algorithm for clustering.
   - Implement a parallelized version of K-means using Python's `multiprocessing` library to improve computational performance.
3. **Performance Evaluation**: Measure and compare the runtime of both implementations and calculate the speedup achieved with parallel processing.


#### **Files**
1. **`data_preprocessing.py`**: Contains the preprocessing pipeline for loading and cleaning the dataset. This includes handling missing values, removing outliers, and standardizing numerical features.
2. **`non_parallel_kmeans.py`**: Implements the standard non-parallel K-means clustering algorithm.
3. **`parallel_kmeans.py`**: Implements the parallelized K-means clustering algorithm using Python's `multiprocessing` library.
4. **`benchmark.py`**: Runs the non-parallel and parallel implementations of K-means, measures runtime for various process counts, and generates a speedup chart to compare performance.
5. **Dataset**: Craigslist used vehicle dataset, scraped from listings across the United States.

#### **Key Features of the Implementation**
- **Parallel K-means**: Parallelized the centroid update and distance calculation processes to utilize multiple CPU cores for faster execution.
- **Speedup Evaluation**: Evaluated the performance gain of the parallel implementation over the non-parallel version using different numbers of processes.

#### **Steps to Reproduce**
1. Ensure you have the dataset file (`data.csv`) in the project directory.
2. Run the following scripts in order:
   - `data_preprocessing.py`: Preprocess the dataset and prepare it for clustering.
   - `benchmark.py`: Execute K-means clustering and compare the performance of parallel and non-parallel implementations.
3. The results, including speedup charts, will be displayed as output.

#### **Dependencies**
- Python 3.7+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

#### **Results**

The project benchmarks revealed that the parallel K-means implementation achieved significant speedups compared to the non-parallel version. However, as shown in the speedup chart, the performance gains plateaued as the number of processes increased, indicating diminishing returns with higher levels of parallelism.
![speedUp](https://github.com/user-attachments/assets/1f6a3a57-3fc9-47b8-9c74-6d79dafbc9fa)

#### **Conclusion**
The parallel K-means clustering implementation demonstrates clear performance advantages over the non-parallel approach, with speedups observed up to **4x** depending on the number of processes and the dataset size. However, parallelization beyond 8 processes did not yield further meaningful improvements due to CPU and memory constraints.
