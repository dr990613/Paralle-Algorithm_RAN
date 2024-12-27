import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset:
    1. Load data from CSV file
    2. Retain numerical features
    3. Handle missing values
    4. Detect and remove outliers using Z-score
    5. Standardize the data using StandardScaler
    6. Return the processed data
    """
    # Load data
    data = pd.read_csv(file_path)

    # Retain only numerical features
    numerical_features = ['year', 'odometer', 'price', 'lat', 'long']
    data = data[numerical_features]

    # Handle missing values: fill with mean
    print("Before handling missing values:")
    print(data.isnull().sum())
    data = data.fillna(data.mean())
    print("After handling missing values:")
    print(data.isnull().sum())

    # Handle outliers: remove using Z-score
    print("Before handling outliers:")
    print(data.describe())
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    print("After handling outliers:")
    print(data.describe())

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Return processed data as a DataFrame
    return pd.DataFrame(data_scaled, columns=numerical_features)


def get_processed_data(file_path):
    """
    A utility function to load and preprocess data, for easy access in other scripts.
    """
    return load_and_preprocess_data(file_path)
