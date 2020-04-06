# Author: Hamidreza Nademi

from ldaClassifier import main as lda
from gmmClassifier import main as gmm
from bayesianClassifier import main as bayesian
from knnClassifier import main as knn
from preprocessing import preprocess_images

# pre process and load data
train_images_dic, test_images_dic = preprocess_images()

# LDA Classifier
lda(train_images_dic, test_images_dic)

# KNN Classifier
knn(train_images_dic, test_images_dic) 

# GMM Classifier
gmm(train_images_dic, test_images_dic)

# Bayesian Classifier
bayesian(train_images_dic, test_images_dic)
