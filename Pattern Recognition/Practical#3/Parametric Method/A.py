# Author: Hamidreza Nademi

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce


def generate_sample(mean, covariance_matrix, n):
    """ generavariancee sample bay given mean and covariance_matrix """

    # x, y = np.random.multivariate_normal(
    #     mean, covariance_matrix, n).T

    generated_samples = np.random.multivariate_normal(
        mean, covariance_matrix, n)

    return generated_samples


def ml_estimation(samples):
    """ compute mean and covariance matrix with ML estimation for given data """

    mean, covariance_matrix = [], np.zeros((2, 2), dtype=float)
    x, y = samples.T

    # calculate mean
    mean.extend([
        reduce((lambda x, y: x+y), x)/len(x),
        reduce((lambda x, y: x+y), y)/len(y)
    ])

    # calculate covariance matrix
    for i in range(len(x)):
        covariance_matrix[0, 0] = covariance_matrix[0, 0] + \
            (x[i]-mean[0])*(x[i]-mean[0])
        covariance_matrix[0, 1] = covariance_matrix[0, 1] + \
            (x[i]-mean[0])*(y[i]-mean[1])
        covariance_matrix[1, 0] = covariance_matrix[1, 0] + \
            (y[i]-mean[1])*(x[i]-mean[0])
        covariance_matrix[1, 1] = covariance_matrix[1, 1] + \
            (y[i]-mean[1])*(y[i]-mean[1])
    covariance_matrix = covariance_matrix/len(x)

    return mean, covariance_matrix


def plot_density(samples, figure_title):

    x, y = samples.T
    x, y = np.meshgrid(x, y)

    figure = plt.figure()
    ax = figure.gca(projection='3d')

    samples = samples.T
    kernel = stats.gaussian_kde(samples)
    z = kernel(np.array([x.ravel(), y.ravel()]))
    z = z.reshape(len(x), len(x))

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    linewidth=0, antialiased=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('p(x)')

    plt.title(figure_title)


def compute_bias(estimated_values, true_value):
    x, y = zip(*estimated_values)
    # compute bias for each dimension
    return [np.mean(x)-true_value[0], np.mean(y)-true_value[1]]


def compute_variance(estimated_values):
    def copmution(dataset):
        # compute sigma x**2
        x_power_two = list(map((lambda x: x**2), dataset))
        sigma_x_power_two = reduce((lambda x, y: x+y), x_power_two)

        variance = (sigma_x_power_two/len(dataset)) - \
            (np.mean(dataset))**2

        return variance

    x, y = zip(*estimated_values)
    return [copmution(dataset=x), copmution(dataset=y)]


def ML_Parameter_Estimate(n, mean_vector, covariance_matrix):
    ##############################
    # b , c
    samples = generate_sample(
        mean=mean_vector,
        covariance_matrix=covariance_matrix,
        n=n
    )
    estimated_mean, estimated_covariance = ml_estimation(samples)

    print(
        f'estimated mean: {estimated_mean} \nestimated covariance matrix:\n {estimated_covariance}'
    )
    plot_density(samples, f'True Density for N={n}')

    samples2 = generate_sample(
        mean=estimated_mean,
        covariance_matrix=estimated_covariance,
        n=10
    )
    plot_density(samples2, f'Estimated Density for N={n}')

    # plt.show()
    ##############################
    ##############################
    # d
    list_etsimated_mean = []
    list_etsimated_covariance = []
    for _ in range(20):
        samples = generate_sample(
            mean=mean_vector,
            covariance_matrix=covariance_matrix,
            n=n
        )
        estimated_mean, estimated_covariance = ml_estimation(samples)

        # select main diameter of covariance_matrix
        estimated_covariance = [
            estimated_covariance[0, 0], estimated_covariance[1, 1]]

        # append new estimated values to list
        list_etsimated_mean.append(estimated_mean)
        list_etsimated_covariance.append(estimated_covariance)
    # show result in output
    print(
        f"""
{'*'*10} N={n}, mean vector={mean_vector}, covariance matrix= {covariance_matrix} {'*'*10}    
Mean:\nbias:{compute_bias(estimated_values= list_etsimated_mean,true_value=[0,5])},
variance:{compute_variance(estimated_values=list_etsimated_mean)}

Covariance matrix:\nbias:{compute_bias(estimated_values= list_etsimated_covariance,true_value=[1,2])}, 
variance:{compute_variance(estimated_values=list_etsimated_covariance)}
""")

    ##############################
# e
ML_Parameter_Estimate(
    n=10,
    mean_vector=[0, 5],
    covariance_matrix=[[1, 1], [1, 2]]
)

# ML_Parameter_Estimate(
#     n=100,
#     mean_vector=[0, 5],
#     covariance_matrix=[[1, 1], [1, 2]]
# )

# ML_Parameter_Estimate(
#     n=1000,
#     mean_vector=[0, 5],
#     covariance_matrix=[[1, 1], [1, 2]]
# )

# f
# ML_Parameter_Estimate(
#     n=10,
#     mean_vector=[5, 0],
#     covariance_matrix=[[1, -1], [-1, 4]]
# )

# ML_Parameter_Estimate(
#     n=100,
#     mean_vector=[5, 0],
#     covariance_matrix=[[1, -1], [-1, 4]]
# )

# ML_Parameter_Estimate(
#     n=1000,
#     mean_vector=[5, 0],
#     covariance_matrix=[[1, -1], [-1, 4]]
# )

plt.show()
input()
