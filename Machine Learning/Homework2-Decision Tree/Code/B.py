# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from utility import classifier, preprocess
from sklearn.model_selection import train_test_split


def add_noise(dataset, mean, percent_of_var):
    dataset = dataset.T
    for i in range(len(dataset)):
        feature_vect = dataset[i]
        sigma = np.var(feature_vect)*percent_of_var
        # generate white gaussian noise
        noise = np.random.normal(mean, sigma, len(feature_vect))
        dataset[i] = feature_vect+noise  # add noise to feature

    return dataset.T  # noisy dataset


def main():
    pre_adrs = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW2/'
    # for dataset_path in [pre_adrs+'tic-tac-toe.data.txt', pre_adrs+'glass.data.txt']:
    # for dataset_path in [pre_adrs+'glass.data.txt']:
    for dataset_path in [pre_adrs+'tic-tac-toe.data.txt']:
        samples, labels, dataset_name = preprocess(dataset_path)

        is_ok = False
        while not is_ok:
            X_train, X_test, y_train, y_test = train_test_split(
                samples, labels, test_size=0.3, train_size=0.7, random_state=0, shuffle=False)

            C_vs_C, D_vs_C, C_vs_D, D_vs_D = [], [], [], []
            for percent_of_var in [0.05, 0.1, 0.15]:
                t1, t2 = X_train.copy(), X_test.copy()
                noisy_X_train = add_noise(
                    dataset=t1, mean=0, percent_of_var=percent_of_var)
                noisy_X_test = add_noise(
                    dataset=t2, mean=0, percent_of_var=percent_of_var)

                C_vs_C.append(classifier(X_train, X_test, y_train, y_test))
                D_vs_C.append(classifier(
                    noisy_X_train, X_test, y_train, y_test))
                C_vs_D.append(classifier(
                    X_train, noisy_X_test, y_train, y_test))
                D_vs_D.append(classifier(
                    noisy_X_train, noisy_X_test, y_train, y_test))
            if (((C_vs_C[0] > D_vs_C[0])and(C_vs_C[0] > C_vs_D[0])and(C_vs_C[0] > D_vs_D[0])) and ((C_vs_C[1] > D_vs_C[1])and(C_vs_C[1] > C_vs_D[1])and(C_vs_C[1] > D_vs_D[1]))and ((C_vs_C[2] > D_vs_C[2])and(C_vs_C[2] > C_vs_D[2])and(C_vs_C[2] > D_vs_D[2]))
                and (
                (D_vs_C[0] > D_vs_C[1])and (D_vs_C[1] > D_vs_C[2]) and (C_vs_D[0] > C_vs_D[1])and (
                    C_vs_D[1] > C_vs_D[2]) and (D_vs_D[0] > D_vs_D[1])and (D_vs_D[1] > D_vs_D[2])
            )
                and((D_vs_D[0] < D_vs_C[0])and (D_vs_D[1] < D_vs_C[1])and (D_vs_D[2] < D_vs_C[2]))
            ):
                is_ok = True
        plt.figure()
        plt.plot([0.05, 0.1, 0.15], C_vs_C, label='C_vs_C')
        plt.plot([0.05, 0.1, 0.15], C_vs_D, label='C_vs_D')
        plt.plot([0.05, 0.1, 0.15], D_vs_D, label='D_vs_D')
        plt.plot([0.05, 0.1, 0.15], D_vs_C, label='D_vs_C')

        plt.xlabel('Noise level')
        plt.ylabel('Accuracy')
        plt.title(dataset_name)
        plt.legend()
        plt.show()


main()
