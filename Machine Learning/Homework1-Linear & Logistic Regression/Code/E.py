# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def model(teta, inpt):
    return teta[0]*inpt[0] + teta[1]*inpt[1]+teta[2]


def sigmoid(x, teta):
    z = np.matmul(a=x, b=teta)
    return 1.0 / (1.0 + np.exp(-z))


def log_likelihood(x, y, teta):
    sigmoid_probs = sigmoid(x, teta)
    return np.sum(y * np.log(sigmoid_probs)
                  + (1 - y) * np.log(1 - sigmoid_probs))


def gradient(x, y, teta):
    sigmoid_probs = sigmoid(x, teta)
    x1, x2, x3 = zip(*x)

    return np.array([[np.sum((y - sigmoid_probs) * x1), np.sum((y - sigmoid_probs) * x2), np.sum((y - sigmoid_probs) * x3)]])


def hessian(x, y, teta):
    sigmoid_probs = sigmoid(x, teta)
    x1, x2, x3 = zip(*x)

    d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * x1)
    d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 * x2)
    d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x3 * x3)
    d4 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * x2)
    d5 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * x3)
    d6 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x3 * x2)

    H = np.array([
        [d1, d4, d5],
        [d4, d2, d6],
        [d5, d6, d3]
    ])
    return H


def newton_method(x, y, iterations, teta):
    def loss_func(h, y):
        return ((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean())/len(x)

    Δl = np.Infinity
    l = log_likelihood(x, y, teta)
    # Convergence Conditions
    δ = .0000000001
    max_iterations = 15
    i = 0
    teta_cost_epoch = []
    while abs(Δl) > δ and i < max_iterations:
        i += 1
        g = gradient(x, y, teta)
        hess = hessian(x, y, teta)
        H_inv = np.linalg.inv(hess)

        Δ = np.dot(H_inv, g.T)
        ΔΘ_1 = Δ[0][0]
        ΔΘ_2 = Δ[1][0]
        ΔΘ_3 = Δ[2][0]

        # Perform our update step
        teta[0] += ΔΘ_1
        teta[1] += ΔΘ_2
        teta[2] += ΔΘ_3

        # Update the log-likelihood at each iteration
        l_new = log_likelihood(x, y, teta)
        Δl = l - l_new
        l = l_new

        teta_cost_epoch.append([i, loss_func(sigmoid(x, teta), y)])

    return teta, teta_cost_epoch


def main():
    # 2
    dataset_path = 'C:/Users/Hamidreza/Desktop/Master/Semester 2/ML/Homeworks/Pure Code/ML/Hw1/ds/dataset3.txt'
    iterations, init_teta, y_pred = 15, [0, 0, 0], []

    dataset = np.loadtxt(dataset_path, delimiter=',')
    x1, x2, y = zip(*dataset)

    x = []  # merge x1, x2
    for i, j in zip(x1, x2):
        # append bias term
        x.append([i, j, 1])

    estimated_theta, teta_cost_epoch = newton_method(np.asarray(
        x), np.asarray(y), iterations, init_teta)  # b
    print(estimated_theta)

    # 3
    iters, costs = zip(*teta_cost_epoch)
    plt.plot(iters, costs)
    plt.title('cost of each Newton method epoch')
    plt.xlabel('epoch')
    plt.ylabel('cost')

    # 4 ,7
    correct_predict = 0
    for i, j, index in zip(x1, x2, range(len(y))):
        r = model(estimated_theta, [i, j])
        if r >= 0:  # sample label is 1
            y_pred.append(1)
            if y[index] == 1:
                correct_predict += 1
        if r < 0:  # sample label is 0
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
    y_values = []
    for i in x1:
        y_values.append(
            -(estimated_theta[2]+estimated_theta[1]*i)/estimated_theta[1]
        )
    plt.figure()
    plt.plot(class_0_x1, class_0_x2, 'x', color='r', label='class 0')
    plt.plot(class_1_x1, class_1_x2, 'x', color='b', label='class 1')
    plt.plot(x1, y_values)

    plt.show()


main()
