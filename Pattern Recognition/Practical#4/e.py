# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt


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

    return train_data


def main():
    # Compute covariance matrix
    covariance_matrix = np.cov(
        read_train_data(path_file='train_Data.txt'),
        rowvar=False
    )

    # Compute eigenvector and eigenvalue of covariance matrix
    eigenvalues = np.linalg.eigvals(covariance_matrix)

    # Plot d versus eigenvalues
    plt.plot([i for i in range(len(eigenvalues))], eigenvalues)
    plt.show()

    print()


main()
