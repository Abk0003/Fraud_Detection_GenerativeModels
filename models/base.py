import numpy as np

class GenerativeClassifier:
    """
    Base class for generative binary classifiers.
    Y = 1 -> fraud
    Y = 0 -> legitimate
    """

    def fit(self, X, y,alpha=1.0):
        """
        Learn class-conditional distributions P(X | Y)
        and class prior P(Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        #BASIC VALIDATION
        if X.ndim != 2 :
            raise ValueError("X must be 2 dimensional")
        if y.ndim != 1 :
            raise ValueError("y must be 1 dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        #ENFORCING BINARY LABELS
        unique = np.unique(y)
        if not np.all(np.isin(y,[0,1])) :
            raise ValueError("y must contain only 0 or 1")
        #CLASS PRIORS WITH LAPLACE SMOOTHING
        n_samples = y.shape[0]
        n_class_0 = np.sum(y==0)
        n_class_1 = np.sum(y == 1);

        self.class_prior_ = {
            0: (n_class_0 + alpha)/(n_samples + 2*alpha),
            1: (n_class_1 + alpha)/(n_samples + 2*alpha)
        }
        self.n_features_ = X.shape[1]
        self._fit_likelihood(X,y)

        return self

    def _fit_likelihood(self, X, y, alpha=1.0):
        """
        Placeholder: subclasses implement this to fit P(X|Y)
        """
        raise NotImplementedError

    def log_likelihood(self, X, y_class):
        """
        Compute log P(X | Y = y_class).

        Returns
        -------
        log_probs : np.ndarray of shape (n_samples,)
        """
        raise NotImplementedError

    def predict_log_odds(self, X):
        """
        Compute log-odds:
        log P(Y=1 | X) - log P(Y=0 | X)
        """
        log_p_x_y1 = self.log_likelihood(X, y_class=1)
        log_p_x_y0 = self.log_likelihood(X, y_class=0)

        log_prior_odds = np.log(self.class_prior_[1]) - np.log(self.class_prior_[0])

        return log_p_x_y1 - log_p_x_y0 + log_prior_odds

    def predict_proba(self, X):
        """
        Convert log-odds to probability via sigmoid.
        """
        log_odds = self.predict_log_odds(X)
        return 1.0 / (1.0 + np.exp(-log_odds))
