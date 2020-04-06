# Author: Hamidreza Nademi

import numpy as np
import math
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


def compute_euclidean_distance(p, q):
    total = 0
    for i in range(len(p[0])):
        total += (p[0][i]-q[0][i])**2
    return math.sqrt(total)


def main(train_images_dic, test_images_dic):
    # train_images_dic, test_images_dic = preprocess_images()
    s_w = None
    s_b = 0
    total_mean = None
    calsses_info = []

    for key, value in train_images_dic.items():
        calsses_info.append(compute_mean_cov(value))

    lst_cov, lst_mean = zip(*calsses_info)

    # convert tuple to list
    lst_cov = list(lst_cov)
    lst_mean = list(lst_mean)

    s_w = sum(lst_cov)

    total_mean = sum(lst_mean)/len(lst_mean)

    for i in range(len(calsses_info)):
        s_b += 2000*(lst_mean[i]-total_mean) * \
            (lst_mean[i]-total_mean).transpose()

    w = np.linalg.inv(s_w)*s_b

    # Compute eigenvector and eigenvalue of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(w)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    w = eigenvectors

    predicted_test_num_dict = {}
    for i in range(10):
        predicted_test_num_dict[i] = []

    predicted_train_num_dict = {}
    for i in range(10):
        predicted_train_num_dict[i] = []

    # classification
    # on test data
    for key, test_images in zip(test_images_dic.keys(), test_images_dic.values()):
        for test_img in test_images:

            test_img = np.asarray([test_img])

            distance = []
            for i in range(len(calsses_info)):
                mu = np.asarray([lst_mean[i]])

                distance.append(
                    compute_euclidean_distance(
                        p=np.matmul(a=test_img, b=w.transpose()),
                        q=np.matmul(a=mu, b=w.transpose())
                    )
                )

            predicted_test_num_dict[key].append(
                distance.index(
                    min(distance)
                )
            )
    compute_experimental_result(
        predicted_test_num_dict, classifier='LDA', test_or_train='test')

    # # on train data
    # for key, train_images in zip(train_images_dic.keys(), train_images_dic.values()):
    #     for train_img in train_images:

    #         train_img = np.asarray([train_img])

    #         distance = []
    #         for i in range(len(calsses_info)):
    #             mu = np.asarray([lst_mean[i]])

    #             distance.append(
    #                 compute_euclidean_distance(
    #                     p=np.matmul(a=train_img, b=w.transpose()),
    #                     q=np.matmul(a=mu, b=w.transpose())
    #                 )
    #             )

    #         predicted_train_num_dict[key].append(
    #             distance.index(
    #                 min(distance)
    #             )
    #         )
    # compute_experimental_result(
    #     predicted_train_num_dict, classifier='LDA', test_or_train='train')

# main()
