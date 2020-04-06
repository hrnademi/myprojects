# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
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


def plot_image(mean_vector, train_data, eigenvectors, dimension):
    images = []
    x_bar = train_data-np.array([mean_vector]).T
    eigenvectors = eigenvectors.T

    for d in dimension:
        w = eigenvectors[:d]
        # PCA
        y = np.matmul(a=w, b=x_bar)

        # PCA reconstruction
        y_hat = np.matmul(a=y.T, b=w)+mean_vector

        # Extract 5th sample
        images.append(np.reshape(y_hat[4], (28, 28)))

    return images


def main():
    train_data = read_train_data(path_file='train_Data.txt')
    plt.imshow(np.reshape(train_data[::, 4], (28, 28)))
    print()
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
    dic_value_vector = {}
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
        dic_value_vector[eigenvalue] = eigenvector

    eigenvalues = sorted(eigenvalues, reverse=True)
    new_eigenvectors = []
    for value in eigenvalues:
        new_eigenvectors.append(dic_value_vector[value])

    eigenvector = new_eigenvectors

    # Build images
    images = plot_image(
        mean_vector=mean_vector,
        train_data=train_data,
        eigenvectors=eigenvectors,
        dimension=[1, 10, 50, 250, 784]
    )

    # Plot images
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 1
    for i, img in zip(range(1, columns*rows + 1), images):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    print()


main()


def test():
    a = np.array([
        [1, 2, 3],
        [3, 2, 1],
        [1, 0, -1]
    ])

    print(a[::, 0])

# test()
