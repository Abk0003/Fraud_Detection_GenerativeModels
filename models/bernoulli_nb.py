import numpy as np
from .base import GenerativeClassifier

class BernoulliNB(GenerativeClassifier):
    def _fit_likelihood(self,X,y,alpha=1.0):
        self.feature_probs_ = {}
        for c in [0,1] :
            X_c = X[y==c]
            self.feature_probs_[c] = (alpha + np.sum(X_c,axis = 0))/(2*alpha + np.shape(X_c)[0])
    def log_likelihood(self,X,y_class):
        prob = self.feature_probs_[y_class]
        log_prob = X*np.log(prob) + (1-X)*np.log(1-prob)
        return log_prob.sum(axis=1)





