# Author: Hamidreza Nademi

from utility import batch_gd_two_param, lr_closed_form
import numpy as np
import matplotlib.pyplot as plt


def model(teta, inpt):
    return teta[0] + teta[1]*inpt[1]


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

    # 2
    teta_closed_form = lr_closed_form(np.asarray(x), np.asarray(y))       # a
    teta_batch, _ = batch_gd_two_param(x, y, iterations, learning_rate,
                                       init_teta0, init_teta1, model)  # b
    print(teta_closed_form, teta_batch)

    # 3
    model1, model2 = [-3.8957, 1.1930], [-3.6403, 1.1673]  # from part A

    plt.plot(x, y, 'x')
    plt.plot(x, [teta_closed_form[0]+teta_closed_form[1]
                 * i for i in x], label='Closed-form')
    plt.plot(x, [teta_batch[0]+teta_batch[1]*i for i in x], label='Batch GD')
    plt.plot(x, [model1[0]+model1[1]*i for i in x],
             label='Closed-form model from part A')
    plt.plot(x, [model2[0]+model2[1]*i for i in x],
             label='Batch GD model from part A')

    plt.xlabel('featrue')
    plt.ylabel('output')
    plt.legend()
    plt.show()

    # 4
    print(f"""
    x=6.2
    teta_closed_form: {model(teta=teta_closed_form,inpt=6.2)} \n
    teta_batch: {model(teta=teta_batch,inpt=6.2)} \n
    ################
    x=12.8
    teta_closed_form: {model(teta=teta_closed_form,inpt=12.8)} \n
    teta_batch: {model(teta=teta_batch,inpt=12.8)} \n
    ################
    x=22.1
    teta_closed_form: {model(teta=teta_closed_form,inpt=22.1)} \n
    teta_batch: {model(teta=teta_batch,inpt=22.1)} \n
    ################
    x=30
    teta_closed_form: {model(teta=teta_closed_form,inpt=30)} \n
    teta_batch: {model(teta=teta_batch,inpt=30)} \n
    """)


main()
