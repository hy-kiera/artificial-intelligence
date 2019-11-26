from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import interpolation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show(samples, n):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = plt.figure(figsize=(15, 15/n))
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        img = np.reshape(samples[i], (28,28))
        imgplot = ax.imshow(img, cmap=plt.cm.Greys)
        imgplot.set_interpolation('nearest')
    #   ax.set_title(targets[i])

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

def showSamples(tr, images, n):
    picks = np.random.randint(0, len(images), n)
    samples1 = np.take(tr, picks, axis=0)
    samples2 = np.take(images, picks, axis=0)
    samples = np.vstack((samples1, samples2))
    show(samples, len(samples))

def DrawCMat(y, pred_y, labels=range(10)):
    cmat = confusion_matrix(y, pred_y, labels)
    print("\nconfusion matrix\n", cmat)
    sns.set_palette("husl")
    plt.figure(figsize=(12,6))
    sns.heatmap(cmat, annot=True)
    plt.title("confusion matrix")
    plt.show()

""" Deskew """
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

def duplicate_data(X, y):
    """
    Deskew duplicate data
    """
    deskewed_X = np.array([deskew(img) for img in X])
    deskewed_X = deskewed_X.reshape((-1, 784))

    return deskewed_X, y

def dup_data(D1, D2, new):
    deskewed_D1 = np.array([deskew(img) for img in D1])
    deskewed_D2 = np.array([deskew(img) for img in D2])
    deskewed_new = np.array([deskew(img) for img in new])

    showSamples(new, deskewed_new, 7)

    return deskewed_D1, deskewed_D2, deskewed_new

def preprocess(X_train, X_valid):
    """
    Scaling
    """
    scl = StandardScaler()
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_valid = scl.transform(X_valid)

    return X_train, X_valid