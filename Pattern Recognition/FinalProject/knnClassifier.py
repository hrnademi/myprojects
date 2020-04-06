# Author: Hamidreza Nademi

import numpy as np
from preprocessing import knn_preprocess_images
from evaluateResult import compute_experimental_result
import math


def knn_classifier(samples, x, k):
    def compute_point_distance(x_i, x):
        sum = 0
        for i, _ in zip(range(len(x_i)), x):
            sum += (x_i[i]-x[i])**2
        return math.sqrt(sum)

    def predict_number(k_image):
        k_i = [i*0 for i in range(10)]
        images, labeles = zip(*k_image)

        for label in labeles:
            k_i[label] += 1

        return k_i.index(max(k_i))
    # stores all computed distances
    lst_radious = []

    # find radius
    for x_i in samples:
        lst_radious.append([compute_point_distance(x_i[0], x), x_i[1]])

    # sort list
    lst_radious = sorted(lst_radious)

    # select k-th element of list
    first_k_img = lst_radious[:k]

    return predict_number(k_image=first_k_img)


def main(train_images_dic, test_images_dic):
    samples, test_images_dic = knn_preprocess_images(
        train_images_dic, test_images_dic)
    predicted_test_num_dict = {}


    for k in [3,5]:
        predicted_test_num_dict.clear()

        for key in test_images_dic.keys():
            predicted_test_num_dict[key] = []

        for key in test_images_dic.keys():
            for test_img in test_images_dic[key]:
                predicted_num = knn_classifier(samples=samples, x=test_img[0], k=k)
                predicted_test_num_dict[key].append(predicted_num)
        compute_experimental_result(
            predicted_test_num_dict, classifier=f'{k}-NN', test_or_train='test')
    


# main()
