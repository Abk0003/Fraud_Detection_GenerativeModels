import numpy as np
from .base import GenerativeClassifier

class ExponentialNB(GenerativeClassifier):
    def _fit_likelihood(self,X,y,alpha=1.0):
        self.features = {}
        for c in [0,1]:
            X_c = X[y==c]
            self.features[c] = (np.shape(X_c)[0] + alpha)/(np.sum(X_c,axis=0) + alpha)
    def log_likelihood(self,X,y_class):
        rate = self.features[y_class]
        log_rate = np.log(rate) - rate * X
        return log_rate.sum(axis=1)