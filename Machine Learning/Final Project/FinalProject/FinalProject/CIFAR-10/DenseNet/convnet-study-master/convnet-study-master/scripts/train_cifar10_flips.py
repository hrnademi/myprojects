import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.callbacks import ModelCheckpoint
# from rme.datasets import cifar10
from keras.datasets import cifar10

# from rme.datasets.utils import zca_whitening
from rme.models import vgg
# from rme.models import nin_bn_model, vgg_model,nin,vgg
from rme.callbacks import Step, MetaCheckpoint

import numpy as np

# Script that trains on cifar-10 augmenting data with horizontal flips.
# Also applies GCN and ZCA as preprocessing. Keras current implementation
# of ZCA is buggy so we're hacking the iterator to get our own version.


class MyIterator(Iterator):
    def __init__(self, X, y, batch_size=64, shuffle=False, seed=None):
        self.X = X
        self.y = y
        self.mean = None
        self.whitening = None
        super(MyIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # Get indices
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # Get batch
        batch_x = self.X[index_array].copy()
        batch_y = self.y[index_array]
        # Flip images randomly
        flips = np.nonzero(np.random.choice(2, current_batch_size))
        batch_x[flips] = batch_x[flips][:, :, ::-1, :]

        # Apply preprocessing to images
        if self.mean is None or self.whitening is None:
            raise Exception('You must call fit before iterating.')
        batch_x = self.whiten(batch_x)
        return (batch_x, batch_y)

    def fit(self, X):
        # Compute mean and whitening matrix from dataset statistics
        data_shape = X.shape
        size = data_shape[0]
        X = X.copy()
        if len(data_shape) > 2:
            X = X.reshape((size, -1))
        X=np.array(X,dtype='float64')
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        cov = np.dot(X.T, X)/size
        U, S, V = np.linalg.svd(cov)
        self.whitening = np.dot(np.dot(U, np.diag(1./np.sqrt(S + 1e-6))), U.T)

    def whiten(self, X):
        # Apply whitening
        data_shape = X.shape
        size = data_shape[0]
        if len(data_shape) > 2:
            X = X.reshape((size, -1))
        white = X - self.mean
        white = np.dot(white, self.whitening)
        return np.reshape(white, data_shape)


# Choose where you want to save the model and meta checkpoints
file_name = 'weights/cifar10/vgg_flip2'

# Choose a model here
# model = nin_bn_model()
# model = vgg_model()
model = vgg.model('cifar10')

sgd = SGD(lr=1e-1, momentum=.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step schedule. Half the learning rate each 25 epochs.
steps = [25*i for i in range(1, 10)]
rates = [0.1/2**i for i in range(10)]

schedule = Step(steps, rates, verbose=1)
model_cbk = ModelCheckpoint(file_name + '.h5')
# meta_cbk = MetaCheckpoint(file_name + '.meta', schedule=schedule)
meta_cbk = MetaCheckpoint(file_name + '.meta', verbose=1)

# train_set, _, test_set = cifar10.load('data/cifar10', gcn=True, zca=False)
(train_set, y_train), (test_set, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# datagen = MyIterator(train_set['data'], train_set['labels'], shuffle=True)
datagen = MyIterator(train_set, y_train, shuffle=True)
# Compute the mean and the whitening matrix
# datagen.fit(train_set['data'])
datagen.fit(train_set)
# Apply whitening to the test set
# test_set['data'] = datagen.whiten(test_set['data'])
test_set = datagen.whiten(test_set)

model.fit_generator(datagen,
                    samples_per_epoch=train_set.shape[0],
                    nb_epoch=250, verbose=2,
                    callbacks=[schedule, model_cbk, meta_cbk],
                    validation_data=(test_set, y_test))
