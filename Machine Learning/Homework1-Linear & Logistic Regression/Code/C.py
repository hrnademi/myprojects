# Author: Hamidreza Nademi

import numpy as np
from utility import batch_gd_three_param, lr_closed_form, compute_MSE
import matplotlib.pyplot as plt


def model(teta, inpt):
    return teta[0] + teta[1]*inpt[1] + teta[2]*inpt[2]


def main():
    dataset_path = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/Hw1/ds/dataset2.txt'
    iterations, learning_rate, init_teta0, init_teta1, init_teta2 = 1500, 0.01, 0, 0, 0

    dataset = np.loadtxt(dataset_path, delimiter=',')
    x1, x2, y = zip(*dataset)
    x = []  # merge x1, x2
    for i, j in zip(x1, x2):
        x.append([1, i, j])
    # 1
    teta_closed_form = lr_closed_form(np.asarray(x), np.asarray(
        y))                                            # a
    teta_batch, teta_cost_epoch_batch = batch_gd_three_param(x, y, iterations, learning_rate,
                                                             init_teta0, init_teta1, init_teta2, model)                   # b

    print(teta_closed_form, teta_batch)

    # 2
    iteration, cost_batch = zip(*teta_cost_epoch_batch)
    plt.plot(iteration, cost_batch, label='Batch GD')

    plt.title('cost of batch GD')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.legend()
    plt.show()

    # 3
    print(f"""
    x=[1357,5]
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,1357,5])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,1357,5])} \n
    ################
    x=[2500,4]
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,2500,4])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,2500,4])} \n
    """)


main()
