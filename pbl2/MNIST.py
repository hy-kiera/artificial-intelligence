import os, sys
import struct
import numpy as np
import matplotlib.pyplot as pyplot
from SVM import SGDSVM
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    raw_train = read_idx("./dataset/train-images.idx3-ubyte")
    train_data = np.reshape(raw_train, (60000, 784))
    train_label = read_idx("./dataset/train-labels.idx1-ubyte")

    raw_new = read_idx("./dataset/new1k-images.idx3-ubyte")
    new_data = np.reshape(raw_new, (10000, 784))
    new_label = read_idx("./dataset/new1k-labels.idx1-ubyte")

    raw_test = read_idx("./dataset/test-images.idx3-ubyte")
    test_data = np.reshape(raw_test, (10000, 784))
    test_label = read_idx("./dataset/test-labels.idx1-ubyte")

    print(train_data.shape, train_label.shape, new_data.shape, new_label.shape)
    X_train = np.vstack((train_data, new_data))
    y_train = np.concatenate((np.array(train_label), np.array(new_label)))
    print("new X_train shape : ", X_train.shape, "new y_train shape : ", y_train.shape)
    X_test = np.array(test_data)
    y_test = np.array(test_label)
    print("X_test shape : ", X_test.shape, "y_test shape : ", y_test.shape)

    X_train, X_test = preprocess(X_train, X_test)
    
    svm = SGDSVM(C=1.0, eta=0.001, max_iter=1, tol=1e-3, epoch_size=10, batch_size=200, random_state=1234)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    DrawCMat(y_test, y_pred)

    print(y_pred, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy : ", accuracy)

# def main(training_image=None, training_label=None, test_image):
    # TODO

if __name__ == '__main__':
#     main(training_image=sys.argv[1], training_label=sys.argv[2], test_image=sts.argv[3])
    MNIST_Classify_test()
