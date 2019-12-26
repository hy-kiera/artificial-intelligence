"""
 PBL3
 https://sites.google.com/view/ailab-hyu/courses/2019-2-artificial-intelligence/ai-pbl3
"""

import os, sys
import struct
import numpy as np
import cupy as cp
import inspect
import matplotlib.pyplot as pyplot
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.ndimage import interpolation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import cv2

def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0 * image) / totalImage #mu_x
    m1 = np.sum(c1 * image) / totalImage #mu_y
    m00 = np.sum(np.power((c0-m0), 2) * image) / totalImage #var(x)
    m11 = np.sum(np.power((c1-m1), 2) * image) / totalImage #var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage #covariance(x,y)
    mu_vector = np.array([m0, m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]]) # Do you see a similarity between the covariance matrix

    return mu_vector, covariance_matrix

def deskew(image):
    """
    Deskew - affine transform
    """
    c, v = moments(image)
    alpha = v[0,1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    img = interpolation.affine_transform(image, affine, offset=offset)

    return (img - img.min()) / (img.max() - img.min())

def dup_data(data):
    deskewed = np.array([deskew(img) for img in data])
    return deskewed

class SGDSVM(BaseEstimator, ClassifierMixin):
    """A SGD-SVM classifier using GPU"""

    def __init__(self, C=1000.0, eta=0.1, max_iter=5, batch_size=128, random_state=1234):
        """
        Called when initializing the classifier
        Parmeters :
            - C : Penalty parameter C of the error term.
            - eta : Learning Rate.
            - max_iter : Hard limit on iterations within solver, or -1 for no limit.
            - epoch_size : Epoch size for training.
            - batch_size : Mini batch size for SGD(Stochastic Gradient Descent).
        """
        # self.C = C
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def one_hot(self, y):
        """
        one-hot encode : 1 is hot, -1 is cold
        """
        onehot = cp.ones((len(y), self.n_classes))

        classes = cp.tile(cp.arange(0, self.n_classes), len(y)).reshape(len(y), self.n_classes)
        y = y.reshape(-1, 1)
        mask = cp.not_equal(y, classes)
        onehot[mask] = -1
            
        return onehot

    def _SGD(self, X, y, W, b):
        """
        SGD(Stochastic Gradient Descent) by mini batch
        """
        # init
        dW = cp.zeros((self.n_features, self.n_classes)) # (784, 10)###
        db = cp.zeros((1, self.n_classes)) # (1, 10)###

        z = cp.add(cp.dot(X,W), b) 
        conditions = cp.multiply(y,z)
        conditions[conditions <= 1] = 1
        conditions[conditions > 1] = 0

        v = cp.dot(X.T, cp.multiply(-y,conditions))
        dW = cp.add(dW, v)
        db = cp.add(db, cp.multiply(-y,conditions))
        db = cp.sum(db,0)
            
        dW /= self.batch_size
        dW = cp.add(dW, cp.multiply((1 / self.C), W)) # margin
        db /= self.batch_size
        return dW, db

   
    def fit(self, X, y):
        """
        This should fit classifier.
        """
        assert (type(self.C) == float), "C parameter must be float"
        assert (type(self.eta) == float), "eta parameter must be float"
        assert (type(self.max_iter) == int and (self.max_iter >= 1 or self.max_iter == -1)), "max_iter parameter must be positive integer. -1 for no limit"
        assert (type(self.batch_size) == int and self.batch_size >= 1 and self.batch_size <= len(X)), "batch_size parameter must be positive integer"

        self.history_score = list() # for saving score of each epoch

        X = cp.asarray(X)
        y = cp.asarray(y)
        n = X.shape[0] # number of data points
        s = cp.arange(n)
        self.n_features = X.shape[1] # number of features
        self.classes, y_ = cp.unique(y, return_inverse=True)
        self.n_classes = len(self.classes) # number of classes
        y = self.one_hot(y_)

        # init parameter
        self.rgen = cp.random.RandomState(self.random_state) 
        W = self.rgen.normal(loc=0.0, scale=0.01, size=(self.n_features, self.n_classes)) # (784, 10)
        b = cp.ones((1, self.n_classes)) # (1, 10)
        W_ = W[:,:]
        b_ = b[:,:]

        # the best W and b that have the best accuracy
        self.best_W = W_[:]
        self.best_b = b_[:]

        mi = self.max_iter
        
        # SGD Algorithm with Weight Averaging
        for it in range(1, mi+1):
            n_batches = int(cp.ceil(len(y) / self.batch_size)) # number of batches

            # random sampling without replacement
            self.rgen.shuffle(s) # {0, 1, ... , n}

            valid_batch_idx = s[self.batch_size * (n_batches - 1) :]

            X_valid = X[valid_batch_idx]
            y_valid = y_[valid_batch_idx]
            # X_valid = cp.array(X_valid)

            for i in range(n_batches-1):
                # mini-batch
                batch_idx = s[self.batch_size * i : self.batch_size * (i + 1)]
                # gradient
                dw , db = self._SGD(X[batch_idx], y[batch_idx], W, b)

                # update (weight averaging)
                W = cp.subtract(W, cp.multiply(self.eta, dw)) # (784, 10)
                b = cp.subtract(b, cp.multiply(self.eta, db)) # (1, 10)
                W_ = cp.add(cp.multiply((it/(it+1)), W_), cp.multiply((it/(it+1)), W))
                b_ = cp.add(cp.multiply((it/(it+1)), b_), cp.multiply((it/(it+1)), b))


            # keep the best weight
            if self._check_score(X_valid, y_valid, self.best_W, self.best_b) < self._check_score(X_valid, y_valid, W_, b_):
                self.best_W = W_[:]
                self.best_b = b_[:]

            # if it % 100 == 0:
            #     print(f"Iteration {it} / {self.max_iter} \t", end='')
            #     print(f"train_accuracy {accuracy_score(cp.asnumpy(self.predict(X_valid)), cp.asnumpy(y_valid))}")

            # save acc socre of each epoch
            self.history_score.append(self._score(X, y_))

    def get_score_history(self):
        return self.history_score

    def _check_score(self, X, y, W, b):
        """get accuracy score with W"""
        try:
            z = cp.add(np.dot(X, W), b)
            y_pred = cp.argmax(z, axis=1)
            acc = accuracy_score(cp.asnumpy(y), cp.asnumpy(y_pred))
        except AttributeError:
            raise RuntimeError()
            
        return acc

    def _score(self, X, y):
        """get accuracy score"""
        y_pred = self.predict(X)
        acc = accuracy_score(cp.asnumpy(y), cp.asnumpy(y_pred))
        return acc

    def predict(self, X):
        """
        Predict y hat
        """
        if type(X) is not cp.core.core.ndarray:
            X = cp.array(X)
        try:
            z = cp.add(cp.dot(X, self.best_W), self.best_b)
            y_pred = cp.argmax(z, axis=1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return y_pred


def preprocess(X_train, X_valid):
    scl = StandardScaler()
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_valid = scl.transform(X_valid)

    return X_train, X_valid

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=cp.uint8).reshape(shape)

def main(training_image_path, training_label_path, test_image_path):
    #load data
    X_train = read_idx(training_image_path)
    y_train = read_idx(training_label_path)
    X_test = read_idx(test_image_path)

    # deskew -> delite
    deskewed_train = dup_data(X_train)
    kernel = np.ones((2, 2), np.uint8)
    dilated_train = np.array([cv2.dilate(img, kernel, iterations = 1) for img in deskewed_train])
    X_train = np.reshape(dilated_train, (-1, 784))

    deskewed_test = dup_data(X_test)
    dilated_test = np.array([cv2.dilate(img, kernel, iterations = 1) for img in deskewed_test])
    X_test = np.reshape(dilated_test, (-1, 784))

    X_train, X_test = preprocess(X_train, X_test)

    # PCA
    pca = PCA(n_components=0.86)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Polynomial Feature Extraction
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_pca)
    X_test_poly = poly_features.transform(X_test_pca)

    
    # concatenate
    X_train = np.concatenate((X_train, X_train_poly), axis=1)
    X_test = np.concatenate((X_test, X_test_poly), axis=1)

    # SVM
    svm = SGDSVM(C=500.0, eta=0.001, max_iter=500, batch_size=512, random_state=1234)

    # Train
    svm.fit(X_train, y_train)
    # Predict
    y_pred = svm.predict(X_test)

    # save predicted value of test data
    f = open("./prediction.txt", "w")
    out = ""
    for i, val in enumerate(y_pred):
        out += str(val) + "|\n\n"
    # print(out)
    f.write(out)
    f.close()

if __name__ == '__main__':
    main(training_image_path=sys.argv[1], training_label_path=sys.argv[2], test_image_path=sys.argv[3])