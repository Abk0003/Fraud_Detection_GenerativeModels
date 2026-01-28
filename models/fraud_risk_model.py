import numpy as np

class FraudRiskModel:
    def __init__(self, blocks, prior=None):
        """
        blocks: list of RiskBlock objects
        prior: optional class prior override
        """
        self.blocks = blocks
        self.prior = prior

    def fit(self, X, y):
        for block in self.blocks:
            block.fit(X, y)
        return self

    def predict_log_odds(self, X):
        log_odds = sum(block.log_risk(X) for block in self.blocks)

        if self.prior is not None:
            log_odds += np.log(self.prior[1]) - np.log(self.prior[0])

        return log_odds

    def predict_proba(self, X):
        log_odds = self.predict_log_odds(X)
        return 1.0 / (1.0 + np.exp(-log_odds))
