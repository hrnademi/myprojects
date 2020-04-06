# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from utililty import pre_process, plot_data, plot_cluster
from random import uniform, randint
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def main():
    pre_path = 'E:/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW4/'

    inputs, labels = pre_process(pre_path+'Aggregation.txt')
    # 1.
    random_MinPts = [i for i in range(4, 14)]
    random_epsilon = [round(uniform(0.1, 0.3), 2) for _ in range(10)]

    # 2.
    mse_error_lst = np.zeros((len(random_MinPts), len(random_epsilon)))
    models = []  # model parameter and its correspond MSE
    inputs = StandardScaler().fit_transform(inputs)

    for n in range(len(random_epsilon)):
        for m in range(len(random_MinPts)):
            db = DBSCAN(
                eps=random_epsilon[n], min_samples=random_MinPts[m])
            db.fit(inputs)
            y_pred = db.fit_predict(inputs)
            mse_error_lst[n, m] = mean_squared_error(labels, y_pred)
            models.extend(
                [[[random_MinPts[m], random_epsilon[n]], mse_error_lst[m, n], y_pred]])

    figure_3D = plt.figure()
    ax = figure_3D.gca(projection='3d')
    ax.plot_surface(random_MinPts, random_epsilon, mse_error_lst)

    ax.set_xlabel('MinPts')
    ax.set_ylabel('epsilon')
    ax.set_zlabel('MSE')

    _, mse, _ = zip(*models)
    mse = np.asarray(mse)

    idx = mse.argsort()[::-1]
    model_idx = idx[55:100]

    X, Y = zip(*inputs)

    for i in model_idx:
        minPts, epsilon, cluster = models[i][0][0], models[i][0][1], models[i][2]
        cost = models[i][1]
        print(f'cost = {cost}')
        plt.figure()
        plt.scatter(X, Y, c=cluster, cmap="plasma")
        plt.title(f'eps: {epsilon}, MinPts: {minPts}')

    plt.show()


main()
