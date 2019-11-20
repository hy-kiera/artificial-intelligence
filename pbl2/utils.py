from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import interpolation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# class StandardScaler():
#     def __init__(self):
#         pass
    
#     def fit(self, X):
#         self.mean = np.mean(X)
#         self.std = np.std(X)
    
#     def transform(self, X):
#         return (X - self.mean) / self.std
    
#     def fit_trainsform(self, X):
#         self.fit(X)
#         self.transform(X)

# class MinMaxScaler():
#     def __init__(self):
#         pass
    
#     def fit(self, X):
#         self.min = np.min(X)
#         self.max = np.max(X)
    
#     def transform(self, X):
#         return (X - self.min) / (self.max - self.min)
    
#     def fit_trainsform(self, X):
#         self.fit(X)
#         self.transform(X)

def DrawCMat(y, pred_y, labels=range(10)):
    cmat = confusion_matrix(y, pred_y,labels)
    sns.set_palette("husl")
    plt.figure(figsize=(12,6))
    sns.heatmap(cmat, annot=True)
    plt.title("confusion matrix")
    plt.show()

""" Deskew """
def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) # sum of pixels
    m0 = np.sum(c0 * image) / totalImage # mu_x
    m1 = np.sum(c1 * image) / totalImage # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / totalImage # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage # covariance(x,y)
    mu_vector = np.array([m0, m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1] / v[0,0]
    affine = np.array([[1,0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c-np.dot(affine, ocenter)
    img = interpolation.affine_transform(image, affine, offset=offset)
    return (img - img.min()) / (img.max() - img.min())

# deskewed = [deskew(img) for img in trT[1]]
# deskewedTr = np.array(deskewed)

def preprocess(X_train, X_test):
    """
    Preprocessing
        1. Increase data with deskew
        2. Scaling
    """
    # duplicate dataset and deskew it
    # dup_data = data[:,:]
    # deskewed_data = deskew()

    scl = StandardScaler()
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)

    return X_train, X_test