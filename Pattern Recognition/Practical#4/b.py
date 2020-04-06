# Author: Hamidreza Nademi

import numpy as np
import sklearn.decomposition.pca as exp


def read_train_data(path_file):
    """ Read data and build train data matrix """
    def convert_to_float(row):
        """ Convet string to float """
        return [float(i) for i in row]

    train_data = []
    with open(file=path_file, mode='r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            train_data.append(
                line.split('\t')[:784]
            )

    # convert string to float
    train_data = list(map(convert_to_float, train_data))

    return train_data


def propose_suitable_d(eigenvalues):
    """ Propose a suitable d using POV = 95% """

    sum_D = sum(eigenvalues)
    for d in range(0, len(eigenvalues)):
        pov = sum(eigenvalues[:d])/sum_D
        if pov > 0.95:
            return d


def main():
    x = read_train_data(
        path_file='train_Data.txt')

    # Compute covariance matrix
    covariance_matrix = np.cov(
        x,
        rowvar=False
    )

    # Compute eigenvalue of covariance matrix
    eigenvalues = np.linalg.eigvals(covariance_matrix)

    # Propose a suitable d using POV = 95%
    print(f'Suitable d is: {propose_suitable_d(eigenvalues=eigenvalues)}')


main()
