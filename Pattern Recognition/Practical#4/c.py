# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math


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


def compute_mse(mean_vector, train_data, eigenvectors):
    lst_mse = []
    x_bar = train_data-np.array([mean_vector]).T
    eigenvectors = eigenvectors.T

    for d in range(1, len(eigenvectors)):
        w = eigenvectors[:d]
        # PCA
        y = np.matmul(a=w, b=x_bar)

        # PCA reconstruction
        y_hat = np.matmul(a=y.T, b=w)+mean_vector

        # Compute MSE
        lst_mse.append(mean_squared_error(y_true=train_data, y_pred=y_hat.T))

    return lst_mse


def main():
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

    # Compute MSE
    lst_mse = compute_mse(
        mean_vector=mean_vector,
        train_data=train_data,
        eigenvectors=eigenvectors
    )

    # Plot d versus MSE
    plt.plot([i for i in range(1, len(eigenvectors))], lst_mse)
    plt.show()
    print()


main()
