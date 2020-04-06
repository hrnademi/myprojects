# Author: Hamidreza Nademi

import numpy as np
from scipy import signal  # medfilt
import os
import cv2

# Path of train data
train_images = 'C:/Users/Hamidreza/Desktop/Master/Semester 1/Pattern_Recognition/HW/Pure Code/FinalProject/USPS-Data/USPS-Data/Numerals'
test_images = 'C:/Users/Hamidreza/Desktop/Master/Semester 1/Pattern_Recognition/HW/Pure Code/FinalProject/USPS-Data/USPS-Data/Test'


def pca(samples):
    def propose_suitable_d(eigenvalues):
        """ Propose a suitable d using POV = 95% """
        sum_D = sum(eigenvalues)
        for d in range(0, len(eigenvalues)):
            pov = sum(eigenvalues[:d])/sum_D
            if pov > 0.95:
                return d

    def pca_formula(d, train_data, mean_vector, eigenvectors):
        x_bar = train_data-np.array([mean_vector])
        eigenvectors = eigenvectors.T
        w = eigenvectors[:d]

        # PCA formula
        y = np.matmul(a=w, b=x_bar.transpose()).transpose()

        return y

    # Compute mean vector of train data
    mean_vector = np.mean(samples, axis=0)

    # Compute covariance matrix of train data
    covariance_matrix = np.cov(
        np.asarray(samples).transpose(),
        rowvar=True
    )

    # Compute eigenvector and eigenvalue of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute eigenvalue of covariance matrix
    eigenvalues = np.linalg.eigvals(covariance_matrix)

    suitable_d = propose_suitable_d(eigenvalues)

    # PCA
    samples_pca = pca_formula(
        d=suitable_d,
        train_data=samples,
        mean_vector=mean_vector,
        eigenvectors=eigenvectors
    )

    return samples_pca


def preprocess_images():
    """ Resize images and return a dictionary of classified images """
    def pack_list_to_dic(samples, labeles):
        result = {}
        for key in range(10):
            result[key] = []

        for (sample, label) in zip(samples, labeles):
            sample = list(map(lambda i: round(i), sample))
            result[label].append(sample)
        return result

    def unpack_dic_to_list(samples_dict):
        result = []
        for key in samples_dict.keys():
            for value in samples_dict[key]:
                result.append([value, key])
        return result

    train_images_dic = {}
    test_images_dic = {}

    # input image dimensions
    width = 28
    height = 28

    for image_folder in os.listdir(train_images):
        train_images_dic[int(image_folder)] = []
        subfolder_path = train_images+'/'+image_folder

        for train_image in os.listdir(subfolder_path):
            img = cv2.imread(subfolder_path+'/'+train_image)

            # Remove noise
            img = signal.medfilt2d(img[:, :, 0], kernel_size=3)

            # resize image to 15 * 15
            th2 = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_LINEAR)

            # Vectorize image
            res = np.reshape(th2, (1, width*height))
            train_images_dic[int(image_folder)].append(res.tolist()[0])

    for image_folder in os.listdir(test_images):
        test_images_dic[int(image_folder)] = []
        subfolder_path = test_images+'/'+image_folder

        for test_image in os.listdir(subfolder_path):
            img = cv2.imread(subfolder_path+'/'+test_image)

            # resize image to 100 * 100
            res = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)[
                :, :, 0]

            res = np.reshape(res, (1, width*height))
            test_images_dic[int(image_folder)].append(res.tolist()[0])

    # Merge test and train images
    samples_lables = []

    # Un-pack train and test image
    samples_lables.extend(
        unpack_dic_to_list(samples_dict=train_images_dic)+unpack_dic_to_list(samples_dict=test_images_dic))

    samples, labels = list(zip(*samples_lables))

    labels = list(labels)
    samples = list(samples)

    # Perform pca
    samples = pca(samples)

    # Un-pack test and train data to python dictionary
    train_images_dic.clear()
    test_images_dic.clear()

    # 20000-21149 => test image , 0-19999 => train image
    train_images_dic = pack_list_to_dic(samples[:20000], labels[:20000])
    test_images_dic = pack_list_to_dic(
        samples[20000:21500], labels[20000:21500])

    return train_images_dic, test_images_dic


def knn_preprocess_images(train_images_dic, test_images_dic):
    new_train_images_dic = {}
    samples = []
    new_test_images_dic = {}

    for key in train_images_dic.keys():
        new_train_images_dic[key] = []
        new_test_images_dic[key] = []

    for key in train_images_dic.keys():
        for value in train_images_dic[key]:
            new_train_images_dic[key].append([value, key])

    for key in new_train_images_dic.keys():
        samples += new_train_images_dic[key]

    for key in test_images_dic.keys():
        for value in test_images_dic[key]:
            new_test_images_dic[key].append([value, key])

    return samples, new_test_images_dic


# preprocess_images()
