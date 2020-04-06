# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from utility import compute_MSE


def model(teta, inpt):
    return teta[0] + teta[1]*inpt[1]


def closed_form_wlr(q, X, Y, tau):
    xw = X.T * radial_kernel(q, X, tau)
    beta = np.linalg.pinv(xw @ X) @ xw @ Y

    # predict value
    return np.dot(q, beta), beta


def batch_gd_wlr(q, x, y, tau, iterations, learning_rate, init_teta):
    def derivative_wrt_teta1(q, x, y, teta):
        result = 0
        w_i = radial_kernel(q, x, tau)
        for i in range(len(x)):
            result += w_i[i]*x[i][1]*((teta[0] + teta[1]*x[i][1])-y[i])
        result = result/len(x)
        return result

    def derivative_wrt_teta0(q, x, y, teta):
        result = 0
        w_i = radial_kernel(q, x, tau)
        for i in range(len(x)):
            result += w_i[i]*((teta[0] + teta[1]*x[i][1])-y[i])
        result = result/len(x)
        return result

    teta = init_teta
    for _ in range(iterations):
        teta[1] = teta[1]-learning_rate * (derivative_wrt_teta1(q, x, y, teta))
        teta[0] = teta[0]-learning_rate * (derivative_wrt_teta0(q, x, y, teta))
    return np.dot(q, teta), teta


def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))


def main():
    dataset_path = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/Hw1/ds/dataset1.txt'
    iterations, learning_rate, init_teta0, init_teta1 = 1500, 0.01, 0, 0

    dataset = np.loadtxt(dataset_path, delimiter=',')
    x, y = zip(*dataset)
    x = list(x)
    y = list(y)
    outlayer_x, outlayer_y = [], []

    # 1
    outlayer_x.extend(np.arange(6, 8, 0.4))
    outlayer_y.extend(np.arange(20, 25, 1))

    outlayer_x.extend(np.arange(20, 24, 0.8))
    outlayer_y.extend(np.arange(0, 10, 2))

    x.extend(outlayer_x)  # append outlayers data to dataset
    y.extend(outlayer_y)

    x = np.asarray(x)
    x = np.reshape(x, (len(x), 1))

    # append bias term
    x = x.tolist()
    for i in x:
        i.insert(0, 1)
    x = np.asarray(x)
    y = np.asarray(y)

    # 3-a
    tau = 5.0
    closed_form_teta_list, batch_GD_teta_list = [], []
    predict_close, predict_gd = [], []
    for x_i in x:
        y_hat, teta = closed_form_wlr(x_i, x, y, tau)
        closed_form_teta_list.append(teta)
        predict_close.append(y_hat)

    closed_form_teta = np.mean(closed_form_teta_list, axis=0)

    # 3-b
    for x_i in x:
        y_hat, teta = batch_gd_wlr(x_i, x, y, tau, iterations,
                                    learning_rate, [init_teta0, init_teta1])
        batch_GD_teta_list.append(teta)
        predict_gd.append(y_hat)

    batch_GD_teta = np.mean(batch_GD_teta_list, axis=0)
    print(f"{tau}, {closed_form_teta} , {batch_GD_teta} , cost: {compute_MSE(closed_form_teta, x, y, model)} , {compute_MSE(batch_GD_teta, x, y, model)} ")

    closed_form_model_in_part_B = [1.0226, 0.6287]  # [teta0 , teta1]
    batch_GD__model_in_part_B = [0.9862, 0.6320]

    plt.figure()
    plt.plot(x[:, 1], y, 'x')
    plt.plot(x[:, 1], predict_close, 'x')
    plt.xlabel('featrue')
    plt.ylabel('predicted y')
    plt.title('closed-form')

    plt.figure()
    plt.plot(x[:, 1], y, 'x')
    plt.plot(x[:, 1], predict_gd, 'x')
    plt.xlabel('featrue')
    plt.ylabel('predicted y')
    plt.title('Bath GD')

    plt.figure()
    plt.plot(x[:, 1], y, 'x')
    plt.plot(x[:, 1], [closed_form_teta[0]+closed_form_teta[1]
                       * i for i in x[:, 1]], label='Closed-form')
    plt.plot(x[:, 1], [batch_GD_teta[0]+batch_GD_teta[1]
                       * i for i in x[:, 1]], label='Batch GD')
    plt.plot(x[:, 1], [closed_form_model_in_part_B[0]+closed_form_model_in_part_B[1]
                       * i for i in x[:, 1]], label='Closed-form model from part B')
    plt.plot(x[:, 1], [batch_GD__model_in_part_B[0]+batch_GD__model_in_part_B[1]
                       * i for i in x[:, 1]], label='Batch GD model from part B')

    plt.xlabel('featrue')
    plt.ylabel('output')
    plt.legend()
    plt.show()


main()
