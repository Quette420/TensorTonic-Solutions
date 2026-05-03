import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    n_arr = np.array(x)
    return 1/(1 + np.exp(-n_arr))