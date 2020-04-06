# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt


def plot_data(inputs, labels, sample_icon, sample_color, title):
    """ Plot dataset """
    x0, x1 = zip(*inputs)
    uniqe_lbls, dict_data = set(labels), {}

    for label in uniqe_lbls:
        dict_data[label] = []

    for i in range(len(labels)):
        dict_data[labels[i]].append(inputs[i])

    for label, color in zip(dict_data.keys(), sample_color):
        x0, x1 = zip(*dict_data[label])
        plt.plot(x0, x1, sample_icon, color=color, label=f'class {int(label)}')

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()


def plot_cluster(clusters, means, colors, sample_icon, title):
    for cluster, mean, color in zip(clusters, means, colors):
        x0, x1 = zip(*cluster)
        plt.plot(x0, x1, sample_icon, color=color)
        plt.plot(mean[0], mean[1], marker='X', color='BLACK')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)


def pre_process(dataset_path):
    inputs = []
    dataset = np.loadtxt(dataset_path, delimiter=',')
    x0, x1, labels = zip(*dataset)

    for i in range(len(x0)):
        inputs.extend([[x0[i], x1[i]]])

    return inputs, labels
