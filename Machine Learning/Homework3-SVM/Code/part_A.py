# Author: Hamidreza Nademi

import numpy as np
from sklearn import svm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_data(inputs, labels):
    x0, x1 = zip(*inputs)
    class_0_x, class_0_y, class_1_x, class_1_y = [], [], [], []
    for i in range(len(x0)):
        if labels[i] == 1:
            class_1_x.append(x0[i])
            class_1_y.append(x1[i])
        elif labels[i] == 0:
            class_0_x.append(x0[i])
            class_0_y.append(x1[i])

    plt.plot(class_0_x, class_0_y, 'x', color='r', label='class 0')
    plt.plot(class_1_x, class_1_y, 'x', color='b', label='class 1')


def main():
    pre_path = 'E:/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW3/'

    # 1.
    data1 = loadmat(pre_path+'hw3_dataset1.mat')
    inputs, labels = data1['X'], data1['y']
    plot_data(inputs, labels)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()

    # 2.
    for c_param in [0.1, 1,10, 100]:
        clf = svm.SVC(C=c_param, kernel='linear')
        clf.fit(inputs, np.ravel(labels))

        # compute accuracy
        y_pred = clf.predict(inputs)
        confusion_mat = confusion_matrix(labels, y_pred)
        acc = sum(np.diag(confusion_mat)/len(labels))*100

        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        plt.figure()
        # plotting the decision boundary
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plot_data(inputs, labels)  # plot dataset

        plt.xlim(0, 4.5)
        plt.ylim(1.5, 5)
        plt.title(f'C= {c_param}')
        plt.xlabel('x0')
        plt.ylabel('x1')
        print(f'C= {c_param} , accuracy= {acc}')
    plt.show()


main()
