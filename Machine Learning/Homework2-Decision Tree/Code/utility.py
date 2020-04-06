# Author: Hamidreza Nademi

import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def classifier(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    acc = np.mean(cross_val_score(clf, X_train, y_train, cv=10))
    return acc*100


def preprocess(ds_path):
    ds_name = ds_path.split('/')[-1]

    if ds_name == 'glass.data.txt':
        ds = np.loadtxt(ds_path, delimiter=',', dtype=float)
        np.random.shuffle(ds)

        samples = ds[:, 1:10]
        labels = ds[:, 10]

    elif ds_name == 'tic-tac-toe.data.txt':
        ds = np.loadtxt(ds_path, delimiter=',', dtype='<U8')
        # Convert to a numerical dataset
        ds[ds == 'x'], ds[ds == 'b'], ds[ds == 'o'] = 0, 1, 2  # inputs
        ds[ds == 'positive'], ds[ds == 'negative'] = 3, 4     # outputs
        ds = np.asarray(ds, dtype=int)  # change data type to int
        
        np.random.shuffle(ds)   # shuffle dataset
        samples = ds[:, 0:9]
        labels = ds[:, 9]
    return samples, labels, ds_name
