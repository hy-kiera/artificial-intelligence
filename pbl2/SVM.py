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
    global INF
    INF = int(2e31)

    def __init__(self, C=1.0, eta=0.01, max_iter=-1, tol=1e-3, epoch_size=1, batch_size=1):
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
        onehot = np.ones((np.shape(y)[0], self.n_class))
        
        for i in range(self.n_class):
            onehot[:, i][y != i] = -1
            
        return onehot

    def _SGD(self, X, y, k, mode):
        """
        SGD(Stochastic Gradient Descent) by mini batch
        """
        # init
        dW = np.zeros((self.n_features, self.n_class)) # (784, 10)
        db = np.zeros((1, self.n_class)) # (1, 10)

        print("BX shape : ", np.array(X).shape, "By shape : ", np.array(y).shape, "\n")

        for r in range(self.batch_size):
            print("y[r].shape : ", y[r].shape, "X[r].shape : ", X[r,:].shape)
            print("W.T.shape : ", self.W.T.shape, "b.shape : ", self.b.shape, "\n")
            # using chain rule
            # z = np.add(np.dot(X[r,:].reshape(self.n_features, 1), self.W.T), self.b) # (batch_size, 784) * (784, 10).T + (1, 10)
            z = np.add(np.dot(self.W.T, X[r,:]), self.b.T) # (10, 784) * (784,) + (10, 1)
            print("z.shape : ", z.shape, "\n")
            cond = np.dot(y[r].T, z)
            if np.dot(y[r].T, z) <= 1:
                if mode == 'W':
                    v = np.dot(X[r,:], -y[r].T)
                    np.add(dW, v)
                if mode == 'b':
                    np.add(db, -y[r].T)

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

        self.n_features = np.array(X).shape[1]
        self.n_class = len(set(y))

        # init parameter
        self.W = np.random.rand(self.n_features, self.n_class) # (784, 10)
        self.b = np.ones((1, self.n_class)) # (1, 10)

        if self.max_iter == -1:
            mi = INF
        else:
            mi = self.max_iter

        # SGD Algorithm with Weight Averaging
        for epoch in range(self.epoch_size):
            for it in range(1, mi+1):
                n_batchs = int(np.ceil(len(y) / self.batch_size))
                print("batch size : ", self.batch_size)
                print("n_batchs : ", n_batchs)

                # random sampling without replacement
                s = np.arange(n_batchs)
                np.random.shuffle(s)

                for k in s:
                    # mini batch
                    ks = self.batch_size * k
                    ke = self.batch_size * (k + 1)

                    BX =  X[ks : ke]
                    By =  self.one_hot(y)[ks : ke]
                    BX =  np.reshape(BX, (self.batch_size, self.n_features))
                    By =  np.reshape(By, (self.batch_size, self.n_class))

                    print("BX : \n", BX)
                    print("By : \n", By)

                    # update - SAG(stochastic average gradient)
                    update_W = self.W + self.eta * self._SGD(BX, By, k, 'W') # (784, 10)
                    update_b = self.b + self.eta * self._SGD(BX, By, k, 'b') # (1, 10)
                    self.W = (it/it+1) * self.W + (it/it+1) * update_W
                    self.b = (it/it+1) * self.b + (it/it+1) * update_b

                    # compare with tol in order to stop training
                    if np.all(np.abs(self.W - self.W) <= self.tol):
                        break

        # return self

    def predict(self, X):
        """
        Predict y hat
        """
        try:
            # W : (784, 10), b : (1, 10)
            # X : (n, 784), y : (n,)
            X = np.reshape(X, (len(X), self.n_features))
            z = np.dot(X, self.W) + self.b
            y_pred = np.argmax(z, axis=1)

            # y_pred = np.add(np.dot(X, self.W.T), self.b)
            print("y_pred.shape : ", y_pred.shape)
            # y_pred = self.W * X + b # (784, 10) * (n, 784) + (1, 10) = (n, 10)
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return y_pred

    # def score(self, )