import numpy as np

from models import (
    BernoulliNB,
    PoissonNB,
    ExponentialNB,
    GDA,
    RiskBlock,
    FraudRiskModel
)

from decision import flag_fraud


# feature groups (example)
blocks = [
    RiskBlock(BernoulliNB(), capability_features),
    RiskBlock(PoissonNB(), intent_features),
    RiskBlock(ExponentialNB(), impact_features),
    RiskBlock(GDA(), context_features)
]

model = FraudRiskModel(blocks)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)
y_pred = flag_fraud(proba, c_fp=1, c_fn=20)
