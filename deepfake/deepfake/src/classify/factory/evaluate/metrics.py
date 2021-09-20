from sklearn.metrics import log_loss as _log_loss

import numpy as np

# dict key should match name of function

def log_loss(y_true, y_prob, **kwargs):
    return {'log_loss': _log_loss(y_true, y_prob, eps=1e-7)}