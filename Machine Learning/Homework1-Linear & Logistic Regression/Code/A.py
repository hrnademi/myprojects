# Author: Hamidreza Nademi


import numpy as np
from utility import batch_gd_two_param, lr_closed_form, stochastic_gd, compute_MSE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def model(teta, inpt):
    return teta[0] + teta[1]*inpt[1]


def main():
    dataset_path = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/Hw1/ds/dataset1.txt'
    iterations, learning_rate, init_teta0, init_teta1 = 1500, 0.01, 0, 0
    # 1
    dataset = np.loadtxt(dataset_path, delimiter=',')
    x, y = zip(*dataset)
    x = np.asarray(x)
    x = np.reshape(x, (len(x), 1))
    # append bias term
    x = x.tolist()
    for i in x:
        i.insert(0, 1)
    x = np.asarray(x)
    
    teta_closed_form = lr_closed_form(np.asarray(x), np.asarray(y))       # a
    teta_sgd, teta_cost_epoch_sgd = stochastic_gd(
        x, y, iterations, learning_rate, init_teta0, init_teta1, model)               # b
    teta_batch, teta_cost_epoch_batch = batch_gd_two_param(x, y, iterations, learning_rate,
                                                           init_teta0, init_teta1, model)                   # c

    print(teta_closed_form, teta_sgd, teta_batch)

    # 3
    plt.plot(x[:,1], y, 'x')
    plt.plot(x[:,1], [teta_closed_form[0]+teta_closed_form[1]* i for i in x[:,1]], label='Closed-form')
    plt.plot(x[:,1], [teta_sgd[0]+teta_sgd[1]*i for i in x[:,1]], label='Stochastic GD')
    plt.plot(x[:,1], [teta_batch[0]+teta_batch[1]*i for i in x[:,1]], label='Batch GD')

    plt.xlabel('featrue')
    plt.ylabel('output')
    plt.legend()

    # 4
    print(f"""
    x=6.2
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,6.2])} \n
    teta_sgd: {model(teta=teta_sgd,inpt=[1,6.2])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,6.2])} \n
    ################
    x=12.8
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,12.8])} \n
    teta_sgd: {model(teta=teta_sgd,inpt=[1,12.8])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,12.8])} \n
    ################
    x=22.1
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,22.1])} \n
    teta_sgd: {model(teta=teta_sgd,inpt=[1,22.1])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,22.1])} \n
    ################
    x=30
    teta_closed_form: {model(teta=teta_closed_form,inpt=[1,30])} \n
    teta_sgd: {model(teta=teta_sgd,inpt=[1,30])} \n
    teta_batch: {model(teta=teta_batch,inpt=[1,30])} \n
    ################
    """)

    # 6
    iteration, cost_sgd = zip(*teta_cost_epoch_sgd)
    _, cost_batch = zip(*teta_cost_epoch_batch)

    plt.figure()
    plt.plot(iteration, cost_sgd)
    plt.title('online GD')
    plt.xlabel('iteration')
    plt.ylabel('cost')

    plt.figure()
    plt.plot(iteration, cost_batch)
    plt.title('batch GD')
    plt.xlabel('iteration')
    plt.ylabel('cost')

    plt.figure()
    plt.plot(iteration, cost_batch, label='Batch GD')
    plt.plot(iteration, cost_sgd, label='Stochastic GD')

    plt.title('cost of online and batch GD')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.legend()

    # 7
    teta_0_lst = np.arange(-10, 10, 0.1)  # range for teat0
    teta_1_lst = np.arange(-1, 4, 0.1)        # range for teat1

    x_fig, y_fig = np.meshgrid(teta_0_lst, teta_1_lst)
    cost = np.zeros((len(teta_1_lst), len(teta_0_lst)))

    for teta_1, i in zip(teta_1_lst, range(len(teta_1_lst))):
        for teta_0, j in zip(teta_0_lst, range(len(teta_0_lst))):
            cost[i, j] = compute_MSE([teta_0, teta_1], x, y, model)
    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x_fig, y_fig, cost)

    ax.set_xlabel('teta_0')
    ax.set_ylabel('teta_1')
    ax.set_zlabel('J(teta_0,teta_1)')

    plt.show()


main()
