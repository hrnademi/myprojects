# Author: Hamidreza Nademi

import numpy as np


def read_train_data(path_file):
    """ Read data and build train data matrix """
    def convert_to_float(row):
        """ Convet string to float """
        return [float(i) for i in row]

    def vectorize_row(row):
        return [round(i) for i in row]

    train_data = []
    with open(file=path_file, mode='r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            train_data.append(
                line.split('\t')[:784]
            )

    # convert string to float
    train_data = list(map(convert_to_float, train_data))

    # Vectorize each train data
    train_data = list(map(vectorize_row, train_data))

    return np.asarray(train_data).T


def pca(d, train_data, mean_vector, eigenvectors):
    x_bar = train_data-np.array([mean_vector]).T
    eigenvectors = eigenvectors.T
    w = eigenvectors[:d]

    # PCA formula
    y = np.matmul(a=w, b=x_bar)

    return y


# read data
train_data = read_train_data(path_file='train_Data.txt')

# Compute covariance matrix of train data
covariance_matrix = np.cov(
    train_data,
    rowvar=True
)

# Compute mean vector of train data
mean_vector = np.mean(train_data, axis=1)

# Compute eigenvector and eigenvalue of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# PCA
print(
    pca(
        d=int(input('Enter number of features: ')),
        train_data=train_data,
        mean_vector=mean_vector,
        eigenvectors=eigenvectors
    )
)
