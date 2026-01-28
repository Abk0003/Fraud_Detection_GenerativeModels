import numpy as np
from .base import GenerativeClassifier
from scipy.special import gammaln

class PoissonNB(GenerativeClassifier):
    """
    Estimate lambda_j_y for each feature j of class y using smooth MLE
    lambda_j_y = (sum of X_j for class y + alpha)/(n_samples + alpha)
    """
    def _fit_likelihood(self,X,y,alpha=1.0):
        self.feature_rates_ = {}
        for c in [0,1] :
            X_c = X[y==c]
            self.feature_rates_[c] = (alpha + np.sum(X_c,axis=0))/(np.shape(X_c)[0]+alpha)

    def log_likelihood(self,X,y_class):
        rate = self.feature_rates_[y_class]
        log_rate = X * np.log(rate) - rate - np.log(gammaln(X+1))
        return log_rate.sum(axis=1)





