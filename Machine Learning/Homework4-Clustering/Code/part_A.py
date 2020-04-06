# Author: Hamidreza Nademi

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from random import uniform, sample
from utililty import pre_process, plot_data, plot_cluster


def k_means(k, train_set, max_iter):
    x0, x1 = zip(*train_set)
    means, clusters, costs, iteration = [], [], [], 0

    # initialize randomly means
    for _ in range(k):
        means.extend([
            [uniform(min(x0), max(x0)), uniform(min(x1), max(x1))]
        ])
        clusters.append([])

    # K-Means algorithm implementation
    while iteration < max_iter:
        for i in range(len(clusters)):
            clusters[i].clear()
        # E-Step
        for sample in train_set:
            temp_dist = []
            for mean in means:
                temp_dist.append(np.linalg.norm(
                    np.array(sample)-np.array(mean)))
            clusters[temp_dist.index(min(temp_dist))].append(sample)

        # M-Step
        for i in range(len(clusters)):
            means[i] = np.mean(np.array(clusters[i]), axis=0)

        costs.append(k_means_cost_func(means, clusters))

        iteration += 1
    return means, clusters, costs


def k_means_cost_func(means, clusters):
    cost = 0
    for i in range(len(means)):
        for sample in clusters[i]:
            cost += np.linalg.norm(np.array(sample)-np.array(means[i]))
    return cost


def compute_wss(means, clusters):
    s_w, s_b = 0, 0
    for i in range(len(clusters)):
        s_w += np.var(clusters[i])
    # mean = np.mean(means, axis=0)
    # for i in range(len(means)):
    #     s_b += np.linalg.norm(mean-means[i])
    return 1/s_w


def main():
    pre_path = 'E:/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW4/'

    inputs, labels = pre_process(pre_path+'Aggregation.txt')
    plot_data(
        inputs, labels,
        sample_icon='*',
        sample_color=['r', 'g', 'b', 'm', 'y', 'Olive', 'DARKSALMON'],
        title='Orginal Data')

    cost_per_iteration, max_iter, WSS_costs, lst_cluster, lst_mean = [], 15, [], [], []
    # 1, 2, 3
    for k in [i+1 for i in range(10)]:
        means, clusters, cost_per_iteration = k_means(k, inputs, max_iter)
        lst_cluster.append(clusters)
        lst_mean.append(means)
        WSS_costs.append(compute_wss(means, clusters))
        # 4
        plt.figure()
        plt.plot([i+1 for i in range(max_iter)], cost_per_iteration)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title(f'k= {k}')

    # 5
    plt.figure()
    plt.plot([i+1 for i in range(10)], WSS_costs)
    plt.xlabel('K')
    plt.ylabel('WSS')

    # 6
    colors = ['INDIANRED', 'DEEPPINK', 'MEDIUMVIOLETRED', 'DARKORANGE',
              'GOLDENROD', 'MAGENTA', 'DARKSLATEBLUE', 'GREENYELLOW', 'STEELBLUE', 'DARKSLATEGRAY']
    WSS_costs, lst_cluster = np.array(WSS_costs), np.array(lst_cluster)
    idx = WSS_costs.argsort()[::-1]
    cluster_idx = idx[5:10]

    for i in cluster_idx:
        plt.figure()
        class_colors = sample(colors, k=i+1)
        plot_cluster(
            clusters=lst_cluster[i],
            means=lst_mean[i],
            sample_icon='*',
            colors=class_colors,
            title=f'k= {i+1}'
        )

    plt.show()


main()
