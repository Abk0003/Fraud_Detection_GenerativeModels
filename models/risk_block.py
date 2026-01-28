import numpy as np

class RiskBlock:
    def __init__(self,model,feature,weight=1.0):
        self.model = model
        self.feature = feature
        self.weight = weight

    def fit(self,X,y):
        X_block = X[:,self.feature]
        self.model.fit(X_block,y)
        return self

    def log_risk(self,X):
        X_block = X[:,self.feature]
        return self.weight*self.model.predict_log_odds(X_block)
