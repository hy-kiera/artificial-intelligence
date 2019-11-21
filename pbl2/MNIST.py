import os, sys
import struct
import numpy as np
import matplotlib.pyplot as pyplot
from SVM import SGDSVM
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def load_data(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    # if dataset == "training":
    #     fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')

    # elif dataset == "testing":
    #     fname_img = os.path.join(path, 'test-images-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'test-labels-idx1-ubyte')

    # elif dataset == "new1k":
    #     fname_img = os.path.join(path, 'new1k-images-idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'new1k-labels-idx1-ubyte')
        
    # else:
    #     raise Exception("dataset must be 'testing' or 'training'")

    # # Load everything in some numpy arrays
    # with open(fname_lbl, 'rb') as flbl:
    #     magic, num = struct.unpack(">II", flbl.read(8))
    #     lbl = np.fromfile(flbl, dtype=np.int8)

    # with open(fname_img, 'rb') as fimg:
    #     magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    #     img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    # get_img = lambda idx: (lbl[idx], img[idx])

    # # Create an iterator which returns each image in turn
    # for i in range(len(lbl)):
    #     yield get_img(i)

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def MNIST_Classify_test():
    # data = load_data("training", "./dataset/")
    # train_dataset = np.array(data)

    #load data
    raw_D1 = read_idx("./dataset/train-images.idx3-ubyte")
    D1_data = np.reshape(raw_D1, (60000, 784))
    D1_label = read_idx("./dataset/train-labels.idx1-ubyte")

    raw_new = read_idx("./dataset/new10k-images.idx3-ubyte")
    new_data = np.reshape(raw_new, (10000, 784))
    new_label = read_idx("./dataset/new10k-labels.idx1-ubyte")

    raw_D2 = read_idx("./dataset/test-images.idx3-ubyte")
    D2_data = np.reshape(raw_D2, (10000, 784))
    D2_label = read_idx("./dataset/test-labels.idx1-ubyte")

    # train data, valid data
    X_train = np.vstack((D1_data, new_data))
    y_train = np.concatenate((np.array(D1_label), np.array(new_label)))
    X_valid = np.array(D2_data)
    y_valid = np.array(D2_label)

    X_train, X_valid = preprocess(X_train, X_valid, y_train, y_valid)
    print("X_train.shape : {0}, X_valid.shape : {1}".format(X_train.shape, X_valid.shape))
    print("y_train.shape : {0}, y_valid.shape : {1}".format(y_train.shape, y_valid.shape))
    print()
    
    svm = SGDSVM(C=1.0, eta=0.001, max_iter=5, tol=1e-3, epoch_size=1, batch_size=256, random_state=1234)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_valid)

    # TODO : GridSearchCV
    # C=[0.01,0.1,10,100]
    # tol=[0.01,0.1,10,100]
    # eta=[0.0002,0.002,0.02]
    # max_iter=[100,1000]
    # param_grid = [
    #     {'C' : C,
    #     'tol' : tol,
    #     'eta' : eta,
    #     'max_iter' : max_iter
    #     }
    # ]
    # svm = SGDSVM(epoch_size=10, batch_size=200, random_state=1234)
    # param_grid = [{'C':[1.0], 'tol':[1e-3], 'eta':[0.001], 'max_iter':[1]}]
    # gs = GridSearchCV(svm, param_grid, cv = 2)
    # gs = gs.fit(X_train, y_train)
    # print(gs.best_score_)

    DrawCMat(y_valid, y_pred)

    print(y_pred, y_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print("Accuracy : ", accuracy)

# def main(training_image=None, training_label=None, test_image):
    # TODO - only predict test image (need to save weight and bias value and load them)

if __name__ == '__main__':
#     main(training_image=sys.argv[1], training_label=sys.argv[2], test_image=sts.argv[3])
    MNIST_Classify_test()
