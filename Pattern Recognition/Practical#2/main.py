# Author: Hamidreza Nademi

import numpy as np
from math import exp, pi
import csv
from model import Model
from class_info import ClassInfo
from functools import reduce


def create_bayes_model(method, likelihoodfncs, priors=None, cost_marix=None):
    """ Create bayes model """
    model = None
    if method == 'Bayes':
        model = Model(
            method=method,
            likelihoodfncs=likelihoodfncs,
            priors=priors,
            cost_marix=cost_marix
        )
    elif method == 'MAP':
        model = Model(
            method=method,
            likelihoodfncs=likelihoodfncs,
            priors=priors,
            cost_marix=[[0, 1], [1, 0]]
        )
    elif method == 'ML':
        model = Model(
            method=method,
            likelihoodfncs=likelihoodfncs,
            priors=[1, 1],
            cost_marix=[[0, 1], [1, 0]]
        )
    return model


def classify_by_bayes(model, samples):
    """ Classify each sample with model """
    scores_matrix = []
    predicted_labels = []

    for sample, i in zip(samples, range(len(samples))):
        temp = []
        # Calculate probability of x given for each class
        for likelihood in model.likelihoodfncs:
            gaussi = calculate_gaussian_distribution(
                x=sample,
                mean_vector=likelihood[0][0],
                covariance_matrix=likelihood[1],
                dimension=len(likelihood[0][0])
            )
            # likelihood index
            index = model.likelihoodfncs.index(likelihood)
            temp.insert(index, gaussi)
        scores_matrix.append(temp)

        # Decision rule implementation
        # Update score matrix based on selected decision rule
        if model.method == 'Bayes':
            scores_matrix[i][0] = scores_matrix[i][0]*model.priors[0] * \
                (model.cost_matrix[0][1]-model.cost_matrix[1][1])

            scores_matrix[i][1] = scores_matrix[i][1]*model.priors[1] * \
                (model.cost_matrix[1][0]-model.cost_matrix[0][0])
        elif model.method == 'MAP':
            # for j in range(0, len(likelihood)-1):
            for j in range(len(likelihood)-1):
                scores_matrix[i][j] = scores_matrix[i][j]*model.priors[j]
        elif model.method == 'ML':
            pass

    # Predict label of each sample of dataset
    actual_labels = []
    for class_info in model.likelihoodfncs:
        actual_labels.append(class_info[2])

    # Maximum number is the predicted class of each sample
    for row in scores_matrix:
        predicted_labels.append(
            actual_labels[
                row.index(
                    max(row)
                )
            ]
        )
    return scores_matrix, predicted_labels


def calculate_gaussian_distribution(x, mean_vector, covariance_matrix, dimension):
    """ Calculate probability of x given C """

    first_part = 1/(((2*pi)**(dimension/2)) *
                    ((np.linalg.det(covariance_matrix))**(0.5)))

    second_part = exp((-0.5)*np.matmul(np.matmul(
        a=(x-mean_vector).transpose(), b=np.linalg.inv(covariance_matrix)), (x-mean_vector)))

    # Gaussian distribution formula for a vector
    likelihood = first_part*second_part

    return likelihood


def calculate_covariance_matrix(dataset):
    """ Calculate covariance matrix for dataset matrix """
    covariance_matrix = None

    # Calculate covariance matrix
    covariance_matrix = np.cov(dataset, rowvar=False)

    # check Singularity for covariance matrix
    if np.linalg.det(covariance_matrix) == 0.0:
        row = covariance_matrix.shape[0]
        for i in range(row):
            covariance_matrix[i, i] += 0.0001

    return covariance_matrix


def calculate_mean(dataset):
    """ Calculate mean vector for dataset """

    # convert nested list to 2D array
    data_matrix_current_class = np.array(dataset).astype(np.float)

    # calculate mean vector
    mean_vector = np.mean(data_matrix_current_class, axis=0)
    mean_vector = mean_vector.tolist()

    return mean_vector


def convert_to_float(row):
    """ Convet string to float """
    return [float(i) for i in row]


def process_data(dataset_path):
    """ Pre Process data to build data matrix and detect number of classes """

    label_set = set()
    original_labels = []

    # read data from dataset and build data-matrix
    with open(file=dataset_path, mode='r') as data_file:
        csv_reader = list(csv.reader(data_file))

        # convert string to float
        csv_reader = list(map(convert_to_float, csv_reader))

        data_matrix_row_number = len(csv_reader)
        # we know than the last colum is the labal of each sample
        if dataset_path.split('/')[1] == 'glass.data.txt':
            data_matrix_col_number = len(csv_reader[0])-2
        else:
            data_matrix_col_number = len(csv_reader[0])-1

        # Build data matrix
        data_matrix = np.zeros(
            (data_matrix_row_number, data_matrix_col_number))

        # Now fill data matrix and recognize number of classes
        row_number = 0
        for row in csv_reader:
            # Add each label to set to find number of classes
            label_set.add(
                row[-1]      # latest element is Label for each row
            )
            original_labels.append(row[-1])

            # fill data matrix
            if dataset_path.split('/')[1] == 'glass.data.txt':
                data_matrix[row_number] = row[1:len(row)-1]
            elif dataset_path.split('/')[1] == 'tic-tac-toe.csv':
                data_matrix[row_number] = row[0:len(row)-1]

            row_number += 1

        # For each class build a list of its data
        label_dict = {}
        for label in label_set:
            # label_dict.append = []
            label_dict[label] = []

        #  store each class in a seperate list
        for label in label_set:
            for row in csv_reader:
                if row[-1] == label:
                    if dataset_path.split('/')[1] == 'glass.data.txt':
                        # print(row[1:len(row)-1])
                        label_dict[label].append(row[1:len(row)-1])
                    elif dataset_path.split('/')[1] == 'tic-tac-toe.csv':
                        label_dict[label].append(row[0:len(row)-1])

    # calculate mean and covariance for each class
    lst_class_info = []
    for label in label_dict:
        mean_vector = calculate_mean(label_dict[label])
        covariance_matrix = calculate_covariance_matrix(
            label_dict[label]
        )
        lst_class_info.append(
            ClassInfo(
                label=label,
                dataset=label_dict[label],
                mean_vector=mean_vector,
                covariance_matrix=covariance_matrix,
                total=len(csv_reader)
            ))

    # return a list that contains mean vector and covariance matrix of each detected class
    return lst_class_info, data_matrix, original_labels


def calculate_confusion_matrix(actual_labels, predicted_labels):
    """ Calculate confusion matrix """
    # confusion_matrix = []
    confusion_matrix = None

    unique_predicted_list = sorted(list(set(predicted_labels)))
    unique_actual_list = sorted(list(set(actual_labels)))

    confusion_matrix = np.zeros(
        (len(unique_actual_list), len(unique_predicted_list)))

    for actual, i in zip(unique_actual_list, range(len(actual_labels))):
        for actual_label in actual_labels:
            if actual_label == actual:
                if predicted_labels[actual_labels.index(actual_label)] == actual:
                    confusion_matrix[i, unique_actual_list.index(actual)] += 1
                else:
                    confusion_matrix[i, int(unique_predicted_list.index(
                        predicted_labels[actual_labels.index(actual_label)]))] += 1

    return confusion_matrix


def output(model, data_matrix, actual_labels):
    """ Show result of classification """
    scores_matrix, predicted_labels = classify_by_bayes(
        model=model,
        samples=data_matrix
    )

    # Build confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        actual_labels, predicted_labels)
    print('Confusion matrix for '+model.method+':\n', confusion_matrix)

    # Calculate accuracy for each class
    for row, i in zip(confusion_matrix, range(len(confusion_matrix[0]))):
        for j in range(len(confusion_matrix)):
            if (i == j):
                print('Accuracy for label '+str(i)+': \n',
                      confusion_matrix[i, j]/reduce((lambda x, y: x + y), row))


def main(dataset_path):
    # Pre-process data
    list_class_info, data_matrix, actual_labels = process_data(
        dataset_path=dataset_path)

    # Build argumants for create_bayes_model method
    # Mean and covaraiance for each class in dataset
    likelihoodfncns = []

    # Prior of each detected class in dataset
    priors = []
    for detected_class in list_class_info:
        likelihoodfncns.append(
            [detected_class.mean_vector,
                detected_class.covariance_matrix, detected_class.label[0]]
        )
        priors.append(
            detected_class.prior
        )

    # Binary classification
    if len(set(actual_labels)) <= 2:
        output(
            model=create_bayes_model(
                method='Bayes',
                likelihoodfncs=likelihoodfncns,
                priors=priors,
                cost_marix=[[0, 2], [1, 0]]
            ),
            data_matrix=data_matrix,
            actual_labels=actual_labels)
        output(
            model=create_bayes_model(
                method='MAP',
                likelihoodfncs=likelihoodfncns,
                priors=priors,
                cost_marix=[[0, 1], [1, 0]]
            ),
            data_matrix=data_matrix,
            actual_labels=actual_labels)
        output(
            model=create_bayes_model(
                method='ML',
                likelihoodfncs=likelihoodfncns,
                priors=1,
                cost_marix=[[0, 1], [1, 0]]
            ),
            data_matrix=data_matrix,
            actual_labels=actual_labels)
    # Multi-class classification
    elif len(set(actual_labels)) > 2:
        output(
            model=create_bayes_model(
                method='MAP',
                likelihoodfncs=likelihoodfncns,
                priors=priors,
                cost_marix=[[0, 1], [1, 0]]
            ),
            data_matrix=data_matrix,
            actual_labels=actual_labels)
        output(
            model=create_bayes_model(
                method='ML',
                likelihoodfncs=likelihoodfncns,
                priors=1,
                cost_marix=[[0, 1], [1, 0]]
            ),
            data_matrix=data_matrix,
            actual_labels=actual_labels)


# main(dataset_path='dataset/tic-tac-toe.csv')
print('*'*50)
main(dataset_path='E:/glass.data.txt')
