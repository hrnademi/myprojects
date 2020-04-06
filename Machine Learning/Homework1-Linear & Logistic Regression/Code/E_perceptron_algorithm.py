# Author: Hamidreza Nademi
# perceptron learning algorithm

import numpy as np
import matplotlib.pyplot as plt
from utility import feature_normalization
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))


def model(teta, bias, inpt):
    return teta[0]*inpt[0] + teta[1]*inpt[1]+bias


def gd(x, y, iterations, learning_rate, init_teta):
    def loss_func(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    teta, bias, teta_cost_epoch = init_teta, 0, []
    for i in range(iterations):
        x = np.asarray(x)

        # z = np.dot(x, teta)
        z = np.matmul(a=x, b=teta)

        h = sigmoid(z)
        gradient = np.dot(x.transpose(), (h - y)) / len(y)
        teta -= learning_rate*gradient

        gradient_t3 = sum((h - y) / len(y))
        bias -= learning_rate*gradient_t3

        teta_cost_epoch.append([i, loss_func(h, y)])
    return teta, bias, teta_cost_epoch


def main():
    # 1

    # 2
    dataset_path = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/Hw1/ds/dataset3.txt'
    iterations, learning_rate, init_teta, y_pred = 1500, 0.01, [0, 0], []

    dataset = np.loadtxt(dataset_path, delimiter=',')
    x1, x2, y = zip(*dataset)

    x1, x2, y = feature_normalization(
        features=[x1, x2, y])  # feature normalization

    x = []  # merge x1, x2
    for i, j in zip(x1, x2):
        x.append([i, j])

    estimated_theta, bias, teta_cost_epoch = gd(np.asarray(x), np.asarray(
        y), iterations, learning_rate, init_teta)  # b
    print(estimated_theta, bias)
    # 3
    iters, costs = zip(*teta_cost_epoch)
    plt.plot(iters, costs)
    plt.title('cost of GD')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.legend()

    # 4 ,7
    correct_predict = 0
    for i, j, index in zip(x1, x2, range(len(y))):
        r = model(estimated_theta, bias, [i, j])
        g_r = sigmoid(z=r)
        if g_r > 0.5:  # sample label is 1
            y_pred.append(1)
            if y[index] == 1:
                correct_predict += 1
        elif g_r < 0.5:  # sample label is 0
            y_pred.append(0)
            if y[index] == 0:
                correct_predict += 1
    confusion_mat = confusion_matrix(y, y_pred)
    print(confusion_mat)
    print(f'accuracy={sum(np.diag(confusion_mat)/len(y))*100}%')

    # 5
    class_0_x1, class_0_x2, class_1_x1, class_1_x2 = [], [], [], []
    for i in range(len(y)):
        if y[i] == 0:
            class_0_x1.append(x1[i])
            class_0_x2.append(x2[i])
        elif y[i] == 1:
            class_1_x1.append(x1[i])
            class_1_x2.append(x2[i])

    plt.figure()
    plt.plot(class_0_x1, class_0_x2, 'x', color='r', label='class 0')
    plt.plot(class_1_x1, class_1_x2, 'x', color='b', label='class 1')

    plt.xlabel('x1')
    plt.ylabel('x2')

    # 6
    estimated_theta = np.resize(estimated_theta, (2, 1))
    z = np.dot(x, estimated_theta)
    probs = sigmoid(z)

    # xx, yy = np.meshgrid(x1, x2)
    # # probs = probs.reshape(xx.shape)
    probs = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            r = model(estimated_theta, bias, [x1[i], x2[j]])
            g_r = sigmoid(z=r)
            probs[i, j] += r

    plt.show()


main()
