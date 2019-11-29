"""
 PBL2
 https://sites.google.com/view/ailab-hyu/courses/2019-2-artificial-intelligence/ai-pbl3
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import inspect
import numpy as np

class SGDSVM(BaseEstimator, ClassifierMixin):
    """A SGD-SVM classifier"""

    # infinity number for max_iter
    INF = int(2e30)

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
        onehot = np.ones(self.n_classes)
        
        for i in range(self.n_classes):
            if y != i:
                onehot[i] = -1
            
        return onehot

    def _SGD(self, X, y, W, b, batch_idx):
        """
        SGD(Stochastic Gradient Descent) by mini batch
        """
        # init
        dW = np.zeros((self.n_features, self.n_classes)) # (784, 10)
        db = np.zeros((1, self.n_classes)) # (1, 10)

        for idx in batch_idx:
            Xr = X[idx,:].reshape(self.n_features, 1)
            yr = (self.one_hot(y[idx])).reshape(self.n_classes, 1)

            # using chain rule
            z = np.add(np.dot(W.T, Xr), b.T) 
           
            conditions = np.multiply(yr,z)
            # misclassified
            conditions[conditions <= 1] = 1
            conditions[conditions > 1] = 0
            
            v = np.dot(Xr, np.multiply(-yr.T,conditions.T))
            dW = np.add(dW, v)
            db = np.add(db, np.multiply(-yr.T,conditions.T))
            
        dW /= self.batch_size
        dW = np.add(dW, np.dot((1 / self.C), W)) # margin
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

        n = np.array(X).shape[0] # number of data points
        ss = np.arange(n)
        self.n_features = np.array(X).shape[1] # number of features
        self.n_classes = len(np.unique(y)) # number of classes

        # init parameter
        self.rgen = np.random.RandomState(self.random_state) 
        W = self.rgen.normal(loc=0.0, scale=0.01, size=(self.n_features, self.n_classes)) # (784, 10)
        b = np.ones((1, self.n_classes)) # (1, 10)
        W_ = W[:,:]
        b_ = b[:,:]

        # the best W and b that have the best accuracy
        self.best_W = W_[:]
        self.best_b = b_[:]

        if self.max_iter == -1:
            mi = INF
        else:
            mi = self.max_iter
        
        # SGD Algorithm with Weight Averaging
        for it in range(1, mi+1):
            n_batches = int(np.ceil(len(y) / self.batch_size)) # number of batches

            # random sampling without replacement
            s = np.arange(n_batches) # {0, 1, ... , n_batches-1}
            self.rgen.shuffle(ss) # {0, 1, ... , n}

            X_valid = X[n - self.batch_size :]
            y_valid = y[n - self.batch_size :]

            for i in range(n_batches-1):
                # mini-batch
                batch_idx = ss[self.batch_size * i : self.batch_size * (i + 1)]
                # gradient
                dw , db = self._SGD(X, y, W, b, batch_idx)

                # update (weight averaging)
                W = np.subtract(W, np.multiply(self.eta, dw)) # (784, 10)
                b = np.subtract(b, np.multiply(self.eta, db)) # (1, 10)
                W_ = np.add(np.multiply((it/(it+1)), W_), np.multiply((it/(it+1)), W))
                b_ = np.add(np.multiply((it/(it+1)), b_), np.multiply((it/(it+1)), b))


            # keep the best weight
            if self._check_score(X_valid, y_valid, self.best_W, self.best_b) < self._check_score(X_valid, y_valid, W_, b_):
                self.best_W = W_[:]
                self.best_b = b_[:]

            if it % 10 == 0:
                print(f"Iteration {it} / {self.max_iter} \t", end='')
                print(f"train_accuracy {accuracy_score(self.predict(X), y)}")

            # save acc socre of each epoch
            self.history_score.append(self._score(X, y))

    def get_score_history(self):
        return self.history_score

    def _check_score(self, X, y, W, b):
        """get accuracy score with W"""
        try:
            z = np.add(np.dot(X, W), b)
            y_pred = np.argmax(z, axis=1)
            acc = accuracy_score(y, y_pred)
        except AttributeError:
            raise RuntimeError()
            
        return acc

    def _score(self, X, y):
        """get accuracy score"""
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        return acc

    def predict(self, X):
        """
        Predict y hat
        """
        try:
            z = np.dot(X, self.best_W) + self.best_b
            y_pred = np.argmax(z, axis=1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return y_pred