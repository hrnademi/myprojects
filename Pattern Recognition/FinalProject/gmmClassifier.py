# Author: Hamidreza Nademi

import numpy as np
import math
from preprocessing import preprocess_images
from evaluateResult import compute_experimental_result


def compute_gaussian_distribution(x, cov, mean):

    # _, logdet = np.linalg.slogdet(cov)
    # # det = math.sqrt(logdet)
    # det = math.sqrt(abs(logdet))
    det = math.sqrt(np.linalg.det(cov))

    first_part = 1/((2*np.pi)**(len(x)/2)*(det))
    second_part = math.exp((-0.5)*np.matmul(a=np.matmul(a=(x -
                                                           mean).transpose(), b=np.linalg.inv(cov)), b=x-mean))
    likelihood = first_part*second_part

    return likelihood


def checkConvergency(old_teta, new_teta, dataset):
    """ check is GMM in its best point or not  """
    def estimate_log_likelihood(teta, dataset):
        """ Estimate log likelihood """
        log_likelihood = 0
        for train_data in dataset:
            _, logdet = np.linalg.slogdet(teta[2])
            det = logdet
            det = abs(logdet)

            first = (
                teta[0])/(math.sqrt(((2*math.pi**len(teta[1]))*det)))

            second_part = math.exp((-0.5)*np.matmul(a=np.matmul(a=(train_data -
                                                                   teta[1]).transpose(), b=np.linalg.inv(teta[2])), b=train_data-teta[1]))

            result = first*second_part
            log_likelihood += (result)

        return log_likelihood

    if new_teta == None:
        return False

    dataset = np.asarray(dataset)
    new_likelihood = estimate_log_likelihood(teta=new_teta, dataset=dataset)
    old_likelihood = estimate_log_likelihood(teta=old_teta, dataset=dataset)

    if abs(new_likelihood-old_likelihood) <= (1e-6):
        return True
    else:
        return False


def train_model(train_dataset):
    def compute_cov(dataset):
        """ Calculate covariance matrix for dataset matrix """
        covariance_matrix = None
        # Calculate covariance matrix
        covariance_matrix = np.cov(dataset, rowvar=False)

        # check Singularity for covariance matrix
        if np.linalg.det(covariance_matrix) == 0.0:
            row = covariance_matrix.shape[0]
            for i in range(row):
                covariance_matrix[i, i] += 0.0001

        return covariance_matrix

    def compute_mean(dataset):
        mean_vector = np.mean(dataset, axis=0)
        return mean_vector

    new_teta_list = []
    # temp
    old_teta_list = []
    max_iter = 5

    # for each class compute mean and cov
    for dataset in train_dataset.values():
        dataset = np.asarray(dataset)

        # initialize teta
        old_teta = [
            len(dataset)/20000,     # prior
            compute_mean(dataset),  # mean vector
            compute_cov(dataset),   # covariance matrix
        ]
        old_teta_list.append(old_teta)
        new_teta = None
        for _ in range(max_iter):
            # Expectation step
            res = []
            n = 0
            for train_data in dataset:
                res.append(compute_gaussian_distribution(
                    x=train_data,
                    cov=old_teta[2],
                    mean=old_teta[1]
                )*old_teta[0])

            # res = np.asarray(res)

            # divide total
            # res = res/res

            # where_are_NaNs = np.isnan(res)
            # res[where_are_NaNs] = 1

            # res = res.tolist()
            n = sum(res)

            # Maximization step
            new_mean = 0
            for train_data, lik in zip(dataset, res):
                new_mean += lik*train_data
            new_mean = new_mean/n

            new_cov = np.zeros((len(new_mean), len(new_mean)))
            res = np.asarray(res)
            for train_data, i in zip(dataset, range(len(dataset))):
                new_cov += res[i]*np.outer(train_data -
                                           new_mean, train_data - new_mean)
            new_cov = new_cov/n

            new_prior = n/len(dataset)

            new_teta = [
                new_prior,
                new_mean,
                new_cov
            ]

            if (checkConvergency(old_teta=old_teta, new_teta=new_teta, dataset=dataset)):
                old_teta = new_teta.copy()
                break
            else:
                old_teta = new_teta.copy()
                new_teta.clear()

        new_teta_list.append(old_teta)

    return new_teta_list


def main(train_images_dic, test_images_dic):
    # train_images_dic, test_images_dic = preprocess_images()
    new_teta = train_model(train_dataset=train_images_dic)

    predicted_test_num_dict = {}
    predicted_train_num_dict = {}
    for i in range(10):
        predicted_test_num_dict[i] = []
        predicted_train_num_dict[i] = []

    # classification
    # on test data
    for key, test_images in zip(test_images_dic.keys(), test_images_dic.values()):
        for test_img in test_images:
            posterior = []
            # compute x given each number
            for prior_mean_cov in new_teta:
                likelihood = compute_gaussian_distribution(
                    x=test_img,
                    cov=prior_mean_cov[2],    # covariance matrix
                    mean=prior_mean_cov[1]    # mean vector
                )
                posterior.append(
                    (likelihood*prior_mean_cov[0])
                )
            predicted_test_num_dict[key].append(
                posterior.index(
                    max(posterior)
                )
            )

    compute_experimental_result(
        predicted_test_num_dict, classifier='GMM', test_or_train='test')

    # on train data
    # for key, train_images in zip(train_images_dic.keys(), train_images_dic.values()):
    #     for test_img in train_images:
    #         posterior = []
    #         # compute x given each number
    #         for prior_mean_cov in new_teta:
    #             likelihood = compute_gaussian_distribution(
    #                 x=test_img,
    #                 cov=prior_mean_cov[2],    # covariance matrix
    #                 mean=prior_mean_cov[1]    # mean vector
    #             )
    #             posterior.append(
    #                 (likelihood*prior_mean_cov[0])
    #             )
    #         predicted_train_num_dict[key].append(
    #             posterior.index(
    #                 max(posterior)
    #             )
    #         )
    # compute_experimental_result(predicted_train_num_dict,classifier='GMM', test_or_train='train')


# main()
