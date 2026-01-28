def decision_threshold(c_fp, c_fn):
    """
    Returns optimal probability threshold
    """
    return c_fp / (c_fp + c_fn)


def flag_fraud(proba, c_fp, c_fn):
    tau = decision_threshold(c_fp, c_fn)
    return (proba >= tau).astype(int)
