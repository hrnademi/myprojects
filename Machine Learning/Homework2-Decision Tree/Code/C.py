# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from utility import classifier, preprocess
from sklearn.model_selection import train_test_split
import random


def main():
    pre_adrs = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW2/'
    for dataset_path in [pre_adrs+'tic-tac-toe.data.txt', pre_adrs+'glass.data.txt']:
        samples, labels, dataset_name = preprocess(dataset_path)

        X_train, X_test, y_train, y_test = train_test_split(
            samples, labels, test_size=0.3, train_size=0.7, random_state=0, shuffle=True)

        size = len(X_train)
        index_list = [i for i in range(size)]
        contradictory_examples, misclassifications, uniqe_labels = [], [], list(set(labels))

        for percent in [0.05, 0.1, 0.15]:
            np.random.shuffle(index_list)
            data_idxs = index_list[:int(percent*size)]
            new_X_train, new_y_train = X_train.copy().tolist(), y_train.copy().tolist()

            # Contradictory examples
            for index in data_idxs:                           # build new train and test set
                new_X_train.append(samples[index])

                lbls = uniqe_labels.copy()
                lbls.remove(labels[index])
                # select random label
                new_y_train.append(random.sample(lbls,  1)[0])

            contradictory_examples.append(                      # compute accuracy
                classifier(np.asarray(new_X_train), X_test, np.asarray(new_y_train), y_test))

            # Misclassifications
            new_y_train = y_train.copy().tolist()
            for index in data_idxs:
                lbls = uniqe_labels.copy()
                lbls.remove(labels[index])
                # Change label
                new_y_train[index] = random.sample(lbls,  1)[0]

            misclassifications.append(                          # compute accuracy
                classifier(X_train, X_test, np.asarray(new_y_train), y_test))
        plt.figure()
        plt.plot([5, 10, 15], contradictory_examples,
                 label='Contradictory examples')
        plt.plot([5, 10, 15], misclassifications, label='Misclassifications')
        plt.xlabel('Noise level')
        plt.ylabel('Accuracy')
        plt.title(dataset_name)
        plt.legend()
        plt.show()


main()
