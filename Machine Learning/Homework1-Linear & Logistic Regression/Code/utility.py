# Author: Hamidreza Nademi

import numpy as np
from random import shuffle, randint


def compute_MSE(teta, x, y, model):
    sigma = 0
    for i in range(len(x)):
        sigma += (model(teta, x[i])-y[i])**2

    cost = (1/(2*len(x)))*sigma
    return cost


def lr_closed_form(x, y):
    """ Linear regression closed-form """

    a = np.linalg.inv(np.matmul(a=x.transpose(), b=x))
    b = x.transpose()
    c = np.matmul(a=a, b=b)
    teta = np.matmul(a=c, b=y).tolist()

    return teta


def batch_gd_two_param(x, y, iterations, learning_rate, init_teta0, init_teta1, model):
    def derivative_wrt_teta1(x, y, teta):
        result = 0
        for x_i, y_i in zip(x, y):
            result += x_i[1]*((teta[0] + teta[1]*x_i[1])-y_i)
        result = result/len(x)
        return result

    def derivative_wrt_teta0(x, y, teta):
        result = 0
        for x_i, y_i in zip(x, y):
            result += (teta[0] + teta[1]*x_i[1])-y_i
        result = result/len(x)
        return result

    teta0, teta1, teta_cost_epoch = init_teta0, init_teta1, []
    for i in range(iterations):
        teta1 = teta1-learning_rate * \
            (derivative_wrt_teta1(x, y, [teta0, teta1]))
        teta0 = teta0-learning_rate * \
            (derivative_wrt_teta0(x, y, [teta0, teta1]))

        teta_cost_epoch.append([i, compute_MSE([teta0, teta1], x, y, model)])
    return [float(teta0), float(teta1)], teta_cost_epoch


def batch_gd_three_param(x, y, iterations, learning_rate, init_teta0, init_teta1, init_teta2, model):
    def derivative_wrt_teta0(x, y, teta):
        result = 0
        for x_i, y_i in zip(x, y):
            result += (teta[0] + teta[1]*x_i[1]+teta[2]*x_i[2])-y_i
        result = result/len(x)
        return result

    def derivative_wrt_teta1(x, y, teta):
        result = 0
        for x_i, y_i in zip(x, y):
            result += ((teta[0] + teta[1]*x_i[1]+teta[2]*x_i[2])-y_i)*x_i[1]
        result = result/len(x)
        return result

    def derivative_wrt_teta2(x, y, teta):
        result = 0
        for x_i, y_i in zip(x, y):
            result += ((teta[0] + teta[1]*x_i[1]+teta[2]*x_i[2])-y_i)*x_i[2]
        result = result/len(x)
        return result

    teta, teta_cost_epoch = [init_teta0, init_teta1, init_teta2], []
    for i in range(iterations):
        teta0 = teta[0]-learning_rate*(derivative_wrt_teta0(x, y, teta))
        teta1 = teta[1]-learning_rate*(derivative_wrt_teta1(x, y, teta))
        teta2 = teta[2]-learning_rate*(derivative_wrt_teta2(x, y, teta))

        teta_cost_epoch.append(
            [i, compute_MSE([teta0, teta1, teta2], x, y, model)])

        teta.clear()
        teta = [teta0, teta1, teta2]

    return teta, teta_cost_epoch


def stochastic_gd(x, y, iterations, learning_rate, init_teta0, init_teta1, model):

    teta0, teta1, teta_cost_epoch, samples, data_size = init_teta0, init_teta1, [
    ], [], len(x)

    for i, j in zip(x, y):
        samples.append([i, j])

    for ite in range(iterations):
        #  select random point of dataset
        # shuffle(samples)
        # rand_point = randint(0, data_size-1)
        # x_i, y_i = x[rand_point], y[rand_point]
        # cost_teta0 = ((teta0+teta1*x_i)-y_i)
        # cost_teta1 = ((teta0+teta1*x_i)-y_i)*x_i
        # teta1 = teta1-learning_rate*(cost_teta1)
        # teta0 = teta0-learning_rate*(cost_teta0)

        # other way
        for i in range(len(x)):
            teta1 = teta1-learning_rate * (((teta0+teta1*x[i][1])-y[i])*x[i][1])
            teta0 = teta0-learning_rate*(((teta0+teta1*x[i][1])-y[i]))

        teta_cost_epoch.append([ite, compute_MSE([teta0, teta1], x, y, model)])

    return [float(teta0), float(teta1)], teta_cost_epoch


def feature_normalization(features):
    """ Mean normalization """
    normalized_features = []
    for feature_vect in features:
        mean = np.mean(feature_vect)
        std = np.std(feature_vect)

        X_norm = (feature_vect - mean)/std
        normalized_features.append(X_norm)

    return normalized_features


# def feature_normalization(features):
#     """min-max normalization"""
#     normalized_features = []
#     for feature_vals in features:
#         normalized = []
#         min_data = min(feature_vals)
#         max_data = max(feature_vals)

#         for value in feature_vals:
#             normalized.append(
#                 (value-min_data) / (max_data-min_data)
#             )
#         normalized_features.append(normalized)

#     return normalized_features[0], normalized_features[1]
