

# author: Hamidreza Nademi
# student number: 9725824

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def plot_figure(array_x, array_y):
    """Plot figure"""
    plt.plot(array_x, array_y, 'x')
    plt.axis('equal')
    print(calculate_covariance(array_x, array_y))
    plt.show()


def calculate_covariance(array_x, array_y):
    """ Calculate covariance by hand"""
    # Question 2:
    k = 1000
    # calculate sum of x
    sum_x = reduce((lambda x, y: x+y), array_x)
    # calculate sum of y
    sum_y = reduce((lambda x, y: x+y), array_y)

    # calculate mean for x and y
    mean_x, mean_y = sum_x/k, sum_y/k

    # create an empty 2*2 matrix for new covariance
    t = np.zeros((2, 2), dtype=float)

    # calculate covariance matrix
    
    for i in range(0, k):
        t[0, 0] = t[0, 0]+(array_x[i]-mean_x)*(array_x[i]-mean_x)
        t[0, 1] = t[0, 1]+(array_x[i]-mean_x)*(array_y[i]-mean_y)
        t[1, 0] = t[1, 0]+(array_y[i]-mean_y)*(array_x[i]-mean_x)
        t[1, 1] = t[1, 1]+(array_y[i]-mean_y)*(array_y[i]-mean_y)
    t = t/k
    return t


def main():
    cov_figure1 = [[1, 0], [0, 1]]  # diagonal covariance
    cov_figure2 = [[1, 0.9], [0.9, 1]]  # diagonal covariance

    mean = [0, 0]  # mean or avreage
    k = 1000  # number of samples to generate

    x_figure1, y_figure1 = np.random.multivariate_normal(
        mean, cov_figure1, k).T
    x_figure2, y_figure2 = np.random.multivariate_normal(
        mean, cov_figure2, k).T

    plot_figure(x_figure1, y_figure1)
    plot_figure(x_figure2, y_figure2)


main()
