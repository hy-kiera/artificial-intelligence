"""
 PBL2
 https://sites.google.com/view/ailab-hyu/courses/2018-2/2018-2-artificial-intelligence/ai-pbl-problem-2
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import inspect
import numpy as np

class SGDSVM(BaseEstimator, ClassifierMixin):
    """A SGD-SVM classifier"""

    # infinity number for max_iter
    INF = int(2e30)

    def __init__(self, C=1.0, eta=0.01, max_iter=-1, tol=1e-3, epoch_size=1, batch_size=1, random_state=1234):
        """
        Called when initializing the classifier
        Parmeters :
            - C : Penalty parameter C of the error term.
            - eta : Learning Rate.
            - max_iter : Hard limit on iterations within solver, or -1 for no limit.
            - tol : Tolerance for stopping criterion.
            - epoch_size : Epoch size for training.
            - batch_size : Mini batch size for SGD(Stochastic Gradient Descent).
        """
        # self.C = C
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def one_hot(self, y):
        onehot = np.ones((np.shape(y)[0], self.n_classes))
        
        for i in range(self.n_classes):
            onehot[:, i][y != i] = -1
            
        return onehot

    def _SGD(self, X, y, k, W, b, mode):
        """
        SGD(Stochastic Gradient Descent) by mini batch
        """
        # init
        dW = np.zeros((self.n_features, self.n_classes)) # (784, 10)
        db = np.zeros((1, self.n_classes)) # (1, 10)

        for r in range(self.batch_size):
            Xr = X[r,:].reshape(self.n_features, 1)
            yr = y[r].reshape(self.n_classes, 1)

            # using chain rule
            z = np.add(np.dot(W.T, Xr), b.T) # (10, 784) * (784,) + (10, 1)

            if np.dot(yr.T, z) <= 1:
                if mode == 'W':
                    v = np.dot(Xr, -yr.T)
                    dW = np.add(dW, v)
                if mode == 'b':
                    db = np.add(db, -yr.T)
            dW = np.add(dW, np.dot((1 / self.C), W))

        if mode == 'W':
            dW /= self.batch_size
            return dW
        if mode == 'b':
            db /= self.batch_size
            return db

    def fit(self, X, y):
        """
        This should fit classifier.
        """
        assert (type(self.C) == float), "C parameter must be float"
        assert (type(self.eta) == float), "eta parameter must be float"
        assert (type(self.max_iter) == int and (self.max_iter >= 1 or self.max_iter == -1)), "max_iter parameter must be positive integer. -1 for no limit"
        assert (type(self.tol) == float), "tol parameter must be float"
        assert (type(self.epoch_size) == int and self.epoch_size >= 1), "epoch_size parameter must be positive integer"
        assert (type(self.batch_size) == int and self.batch_size >= 1 and self.batch_size <= len(X)), "batch_size parameter must be positive integer"

        n = np.array(X).shape[0] # number of data points
        self.n_features = np.array(X).shape[1] # number of features
        self.n_classes = len(np.unique(y)) # number of classes

        # init parameter
        # self.W = np.random.rand(self.n_features, self.n_classes) # (784, 10)
        self.rgen = np.random.RandomState(self.random_state)
        self.W = self.rgen.normal(loc=0.0, scale=0.01, size=(self.n_features, self.n_classes))
        self.b = np.ones((1, self.n_classes)) # (1, 10)

        if self.max_iter == -1:
            mi = INF
        else:
            mi = self.max_iter

        # SGD Algorithm with Weight Averaging
        for epoch in range(self.epoch_size):
            # TODO - iter for와 batch for 순서 어떻게? 바꿔야 하지 않을까
            for it in range(1, mi+1):
                n_batches = int(np.ceil(len(y) / self.batch_size))

                # random sampling without replacement
                s = np.arange(n_batches) # {0, 1, ... , n_batches-1}
                self.rgen.shuffle(s)

                update_W = self.W
                update_b = self.b

                for k in s:
                    # mini batch
                    ks = self.batch_size * k
                    ke = self.batch_size * (k + 1)

                    if ke > n:
                        # the batch size is not a multiple of n
                        BX = np.vstack((X[ks:n], X[0:ke-n]))
                        By = np.vstack((self.one_hot(y)[ks:n], self.one_hot(y)[0:ke-n]))
                    else:
                        BX =  X[ks:ke]
                        By =  self.one_hot(y)[ks:ke]
                    
                    BX =  np.reshape(BX, (self.batch_size, self.n_features))
                    By =  np.reshape(By, (self.batch_size, self.n_classes))

                    # update - SAG(stochastic average gradient)
                    update_W = np.subtract(update_W, np.multiply(self.eta, self._SGD(BX, By, k, update_W, update_b, 'W'))) # (784, 10)
                    update_b = np.subtract(update_b, np.multiply(self.eta, self._SGD(BX, By, k, update_W, update_b, 'b'))) # (1, 10)
                    self.W = np.add(np.multiply((it/(it+1)), self.W), np.multiply((it/(it+1)), update_W))
                    self.b = np.add(np.multiply((it/(it+1)), self.b), np.multiply((it/(it+1)), update_b))

                    # compare with tol in order to stop training
                    if np.all(np.abs(np.subtract(self.W, update_W)) <= self.tol):
                        break

    def predict(self, X):
        """
        Predict y hat
        """
        try:
            X = np.reshape(X, (len(X), self.n_features))
            z = np.dot(X, self.W) + self.b
            y_pred = np.argmax(z, axis=1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return y_pred