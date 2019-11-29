import os, sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from SVM import SGDSVM
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def MNIST_Classify_test():
    #load data
    raw_D1 = read_idx("../dataset/train-images.idx3-ubyte") # (60000, 28, 28)
    D1_data = np.reshape(raw_D1, (60000, 784))
    D1_label = read_idx("../dataset/train-labels.idx1-ubyte")

    raw_new = read_idx("../dataset/new10k-images.idx3-ubyte")
    new_data = np.reshape(raw_new, (10000, 784))
    new_label = read_idx("../dataset/new10k-labels.idx1-ubyte")

    raw_D2 = read_idx("../dataset/test-images.idx3-ubyte")
    D2_data = np.reshape(raw_D2, (10000, 784))
    D2_label = read_idx("../dataset/test-labels.idx1-ubyte")

    raw_train = read_idx("../dataset/newtrain-images.idx3-ubyte")
    train_data = np.reshape(raw_train, (80000, 784))
    train_label = read_idx("../dataset/newtrain-labels.idx1-ubyte")

    # deskew
    # deskewed_D1, deskewed_D2, deskewed_new = dup_data(raw_D1, raw_D2, raw_new)

    # train data, valid data
    # X_train = np.vstack((D1_data, D2_data))
    # deskewed_X_train = np.vstack((deskewed_D1, deskewed_D2))
    # # print(f"X_train.shape : {X_train.shape}, deskewed_X_train.shape : {deskewed_X_train.shape}")
    # X_train = np.vstack((X_train, deskewed_X_train.reshape((-1, 784))))
    
    # y_train = np.concatenate((np.array(D1_label), np.array(D2_label)))
    # y_train = np.concatenate((y_train, y_train))
    
    # X_valid = np.array(new_data)
    # y_valid = np.array(new_label)

    X_train = D2_data[:,:]
    y_train = D2_label[:]

    X_valid = new_data[:,:]
    y_valid = new_label[:]

    print("X_train.shape : {0}, X_valid.shape : {1}".format(X_train.shape, X_valid.shape)) # (70000, 784), (10000, 784)
    print("y_train.shape : {0}, y_valid.shape : {1}".format(y_train.shape, y_valid.shape)) # (70000,), (10000,)
    print()

    svm = SGDSVM(C=300.0, eta=0.01, max_iter=200, batch_size=16, random_state=1234)
    start = time.time()
    svm.fit(X_train, y_train)
    end = time.time()

    print("training time : ", end-start, "sec\n")
    
    acc_scores = svm.get_score_history()

    y_pred = svm.predict(X_valid)

    DrawCMat(y_valid, y_pred)

    print(y_pred, y_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print("Accuracy : ", accuracy)

    x = np.arange(0, len(acc_scores)+1, 25)
    plt.title("acc_score")
    plt.plot(acc_scores)
    plt.xticks(x)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.grid(False)
    plt.show()

def main(training_image_path, training_label_path, test_image_path):
    # train
    raw_train = read_idx(training_image_path)
    train_data = np.reshape(raw_train, (-1, 784))
    train_label = read_idx(training_label_path)

    # test
    raw_test = read_idx(test_image_path)
    test_data = np.reshape(raw_test, (-1, 784))
    
    # duplicate data
    dup_train_data, dup_train_label = duplicate_data(raw_train, train_label)

    X_train = np.vstack((train_data, dup_train_data))
    y_train = np.concatenate((np.array(train_label), np.array(dup_train_label)))
    X_test = test_data[:,:]

    # Scaling
    X_train, X_test = preprocess(X_train, X_test)

    # SVM
    svm = SGDSVM(C=10.0, eta=0.001, max_iter=200, batch_size=200, random_state=1126)
    # training
    svm.fit(X_train, y_train)

    # predict
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
    # main(training_image_path=sys.argv[1], training_label_path=sys.argv[2], test_image_path=sys.argv[3])
    MNIST_Classify_test()
