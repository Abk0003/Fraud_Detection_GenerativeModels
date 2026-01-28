def expected_loss(y_true, y_pred, c_fp, c_fn):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return c_fp * fp + c_fn * fn
