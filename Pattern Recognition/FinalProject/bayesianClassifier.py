# Author: Hamidreza Nademi

import numpy as np
import math
import os
from preprocessing import preprocess_images
from evaluateResult import compute_experimental_result


def compute_mean_cov(dataset):
    """ Calculate covariance matrix for dataset matrix """
    covariance_matrix = None
    # Calculate covariance matrix
    covariance_matrix = np.cov(dataset, rowvar=False)

    # check Singularity for covariance matrix
    if np.linalg.det(covariance_matrix) == 0.0:
        row = covariance_matrix.shape[0]
        for i in range(row):
            covariance_matrix[i, i] += 0.0001

    mean_vector = np.mean(dataset, axis=0)

    return covariance_matrix, mean_vector


def compute_gaussian_distribution(x, cov, mean):

    _, logdet = np.linalg.slogdet(cov)
    # det = math.sqrt(logdet)
    det = math.sqrt(abs(logdet))

    first_part = 1/((2*np.pi)**(len(x)/2)*(det))
    second_part = math.exp((-0.5)*np.matmul(a=np.matmul(a=(x -
                                                           mean).transpose(), b=np.linalg.inv(cov)), b=x-mean))
    likelihood = first_part*second_part

    return likelihood


def main(train_images_dic,test_images_dic):
    # train_images_dic, test_images_dic = preprocess_images()
    likelihoods = []
    score_matrix = []
    predicted_test_num_dict = {}
    predicted_train_num_dict = {}

    for key in test_images_dic.keys():
        predicted_test_num_dict[key] = []
        predicted_train_num_dict[key] = []

    for image_ds in train_images_dic.values():
        likelihoods.append(compute_mean_cov(image_ds))

    # on test data
    for key, test_images in zip(test_images_dic.keys(), test_images_dic.values()):
        for test_img in test_images:
            # compute x given each number
            score_row = []
            for likelihood in likelihoods:
                score_row.append(
                    compute_gaussian_distribution(
                        x=test_img,
                        cov=likelihood[0],
                        mean=likelihood[1]
                    )
                )
            score_matrix.append(score_row)
            predicted_test_num_dict[key].append(
                score_row.index(
                    max(score_row)
                )
            )

    compute_experimental_result(
        predicted_test_num_dict, classifier='Baysian', test_or_train='test')

    # on train data
    # for key, train_images in zip(train_images_dic.keys(), train_images_dic.values()):
    #     for train_img in train_images:
    #         # compute x given each number
    #         score_row = []
    #         for likelihood in likelihoods:
    #             score_row.append(
    #                 compute_gaussian_distribution(
    #                     x=train_img,
    #                     cov=likelihood[0],
    #                     mean=likelihood[1]
    #                 )
    #             )
    #         score_matrix.append(score_row)
    #         predicted_train_num_dict[key].append(
    #             score_row.index(
    #                 max(score_row)
    #             )
    #         )
    # compute_experimental_result(predicted_train_num_dict,classifier='Baysian',test_or_train='train')


# main()
