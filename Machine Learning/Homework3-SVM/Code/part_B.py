# Author: Hamidreza Nademi

import numpy as np
from sklearn import svm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


def plot_data(inputs, labels):
    """ Plot dataset """
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
    plt.plot(class_1_x, class_1_y, 'o', color='b', label='class 1')


def main():
    pre_path = 'E:/Master/Semester 2/ML/Homeworks/Pure Code/ML/HW3/'

    # 1.
    data1 = loadmat(pre_path+'hw3_dataset2.mat')
    inputs, labels = data1['X'], data1['y']
    plot_data(inputs, labels)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()

    # 2.
    lst_vals = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
    best_acc, best_sigam, best_c, clf = 0, 0, 0, None
    for sigam in lst_vals:
        for c_param in lst_vals:
            # train svm classifier by Gaussian kernel
            clf = svm.SVC(C=c_param, gamma=sigam, kernel='rbf')
            clf.fit(inputs, np.ravel(labels))
            acc = np.mean(cross_val_score(clf, inputs, labels, cv=10))
            if acc > best_acc:
                best_acc, best_c, best_sigam = acc, c_param, sigam
    print(round(best_acc, 2), best_c, best_sigam)

    # 4.
    # plotting the decision boundary
    xx, yy = np.meshgrid(np.linspace(inputs[:, 0].min(), inputs[:, 1].max(
    ), num=100), np.linspace(inputs[:, 1].min(), inputs[:, 1].max(), num=100))

    clf = svm.SVC(C=best_c, gamma=best_sigam, kernel='rbf')
    clf.fit(inputs, np.ravel(labels))
    plt.contour(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(
        xx.shape), 1, colors="black")
    plt.xlim(-0.6, 0.3)
    plt.ylim(-0.7, 0.5)

    plot_data(inputs, labels)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(f'Best C= {best_c} and sigma= {best_sigam}')
    plt.show()
    # clf = svm.SVC(C=best_c, gamma=best_sigam)
    # z = clf.predict(xx, yy)

    # 3.
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.3, train_size=0.7, random_state=0, shuffle=True)
    lst_vals = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
    lst_train_var, lst_test_var, lst_test_acc, lst_train_acc = [], [], [], []

    for sigam in lst_vals:
        for c_param in lst_vals:
            # train svm classifier by Gaussian kernel for Train dataset
            clf = svm.SVC(C=c_param, gamma=sigam)
            clf.fit(X_train, np.ravel(y_train))
            train_ten_time_ten_fold_acc = cross_val_score(
                clf, X_train, y_train, cv=10)
            lst_train_acc.append(np.mean(train_ten_time_ten_fold_acc))
            lst_train_var.append(np.var(train_ten_time_ten_fold_acc))

            # train svm classifier by Gaussian kernel for Test dataset
            clf = svm.SVC(C=c_param, gamma=sigam)
            clf.fit(X_test, np.ravel(y_test))
            test_ten_time_ten_fold_acc = cross_val_score(
                clf, X_test, y_test, cv=10)
            lst_test_acc.append(np.mean(test_ten_time_ten_fold_acc))
            lst_test_var.append(np.var(test_ten_time_ten_fold_acc))

    plt.figure()
    plt.plot([i for i in range(1, 65)], lst_test_acc)
    plt.xlabel('C and sigma')
    plt.ylabel('Accuracy')
    plt.title('Test acc')

    plt.figure()
    plt.plot([i for i in range(1, 65)], lst_test_var)
    plt.xlabel('C and sigma')
    plt.ylabel('Variance')
    plt.title('Variance of ten-times-ten-fold cross validation for Test dataset')

    plt.figure()
    plt.plot([i for i in range(1, 65)], lst_train_acc)
    plt.xlabel('C and sigma')
    plt.ylabel('Accuracy')
    plt.title('Train acc')

    plt.figure()
    plt.plot([i for i in range(1, 65)], lst_train_var)
    plt.xlabel('C and sigma')
    plt.ylabel('Variance')
    plt.title('Variance of ten-times-ten-fold cross validation for Train dataset')

    plt.show()

    # 5.
    clf = svm.SVC(C=best_c, gamma=best_sigam, kernel='rbf')
    clf.fit(X_test, np.ravel(y_test))
    acc = np.mean(cross_val_score(clf, X_test, y_test, cv=10))
    print(f'Test accuracy is : {acc}')


main()
