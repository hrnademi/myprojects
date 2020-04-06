# Author: Hamidreza Nademi

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, pi
from functools import reduce


def generate_sample(mean, covariance_matrix, n):
    """ generate sample bay given mean and covariance_matrix """
    generated_samples = []

    samples1_x, samples1_y = np.random.multivariate_normal(
        mean, covariance_matrix, n).T

    for samples1_x, samples1_y in zip(samples1_x, samples1_y):
        generated_samples.append([samples1_x, samples1_y])

    return generated_samples


def plot_density(samples, p_x=None, figure_title=None):

    x, y = zip(*samples)
    x, y = np.meshgrid(x, y)

    x = np.asarray(x)
    y = np.asarray(y)
    samples = np.asarray(samples)

    figure = plt.figure()
    ax = figure.gca(projection='3d')

    samples = samples.T
    if p_x != None:
        # z = np.asarray(p_x)
        # z=z.T
        z = []
        for _ in range(len(x)):
            z.append(p_x)
        z = np.asarray(z)
        # z = z.reshape(len(x), len(x))
    else:
        kernel = stats.gaussian_kde(samples)
        z = kernel(np.array([x.ravel(), y.ravel()]))
        z = z.reshape(len(x), len(x))

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    linewidth=0, antialiased=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('p(x)')

    plt.title(figure_title)


def compute_point_distance(x, y):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def estimate_density_parzen_window(generate_sample, h, dimension):
    # store probability for each pair of [x,y]
    p_x = []

    def compute_density(x):
        """ implement parzen window formula """

        # const part of Parzen-Window
        const_term = 1.0/((len(generated_samples))*(h**dimension))

        # count number of data is in window
        k = 0

        #
        for x_i in generated_samples:
            if(compute_point_distance(x, x_i) < (h/2)):
                # if point is in window, increment k
                k += 1

        # compute KDE and return it
        return const_term*k

    # apply parzen formula for each pair [x,y]
    for x_i in generated_samples:
        p_x.append(compute_density(x_i))

    return p_x


def estimate_density_KNN(generate_sample, k):
    # store probability for each pair of [x,y]
    p_x = []

    def compute_density(x):
        """ implement KNN formula """
        # stores all computed distances
        lst_radious = []

        # find radius
        for x_i in generated_samples:
            lst_radious.append(compute_point_distance(x, x_i))

        # sort list
        lst_radious = sorted(lst_radious)

        # select k-th element of list
        radius = lst_radious[k]

        # compute KDE and return it
        return k/((len(generated_samples))*pi*(radius**2))

    # apply parzen formula for each pair [x,y]
    for x_i in generated_samples:
        p_x.append(compute_density(x_i))

    return p_x


# a
# generate samples with each mean an covariance
samples1 = generate_sample(
    mean=[0, 5],
    covariance_matrix=[[1, 1], [1, 2]],
    n=100
)
samples2 = generate_sample(
    mean=[5, 0],
    covariance_matrix=[[1, -1], [-1, 4]],
    n=100
)

# merge generated samples
generated_samples = samples1+samples2

# plot true density
plot_density(generated_samples, figure_title=f'True Density')

# b
# estimate density via Parzen Window
# for h in [0.2, 0.4, 0.8, 1.6]:
#     p_x = estimate_density_parzen_window(
#         generate_sample=generated_samples,
#         h=h,
#         dimension=len(generated_samples[0])
#     )
#     plot_density(generated_samples, p_x,
#                  figure_title=f'Estimated Density for h={h}')

# c
# estimate density via KNN
for k in [1, 10, 30]:
    p_x = estimate_density_KNN(
        generate_sample=generated_samples,
        k=k
    )
    plot_density(generated_samples, p_x,
                 figure_title=f'Estimated Density for k={k}')


plt.show()
