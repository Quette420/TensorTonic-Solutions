import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    np_array = np.array(A)
    return np.transpose(np_array)
