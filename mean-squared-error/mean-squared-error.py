import numpy as np

def mean_squared_error(y_pred, y_true):
    np_pred = np.array(y_pred)
    np_true = np.array(y_true)

    return (1/np_pred.size) * np.sum((np_pred - np_true) ** 2)
