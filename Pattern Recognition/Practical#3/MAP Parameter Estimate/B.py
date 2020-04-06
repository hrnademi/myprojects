# Author: Hamidreza Nademi

import numpy as np
from scipy import stats
from functools import reduce


def generate_sample(mean_vector, covariance_matrix, n):
    """ generavariancee sample bay given mean and covariance_matrix """

    generated_samples = np.random.multivariate_normal(
        mean_vector, covariance_matrix, n)

    return generated_samples


def estimate_mean(samples, covariance_matrix, mu_mean, mu_cov):
    mu_hat_x, mu_hat_y = None, None

    # compute estimated mean for each dimension
    x, y = zip(*samples)

    for _ in x:
        mu_hat_x = ((covariance_matrix[0]**2)*mu_mean[0] +
                    (mu_cov[0]**2*((reduce((lambda x, y: x+y), x)))) /
                     (((len(samples)*mu_cov[0]**2)+covariance_matrix[0]**2)))
    for _ in y:
        mu_hat_y=((covariance_matrix[1]**2)*mu_mean[1] +
                    (mu_cov[1]**2*(reduce((lambda x, y: x+y), y)))) / (((len(samples)*mu_cov[1]**2)+covariance_matrix[1]**2))

    return [mu_hat_x, mu_hat_y]


for n in [10, 100, 1000]:
    samples=generate_sample(
        mean_vector=[0, 5],
        covariance_matrix=[[1, 1], [1, 2]],
        n=n
    )

    MAP_mu_hat=estimate_mean(
        samples=samples,
        covariance_matrix=[1, 2],
        mu_mean=[0, 0],
        mu_cov=[10, 10])

    print(f"N={n} , mean={MAP_mu_hat}\n")
