import numpy as np
from numba import jit
from numba import float64
from numba import int64
"""
This module contains various implementations of ewma based on sample size
"""

@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(arr_in, window):
    """Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param arr_in : np.ndarray, float64. A single dimenisional numpy array
    :paran window : int64. The decay window, or 'span'
    :return: np.ndarray. The EWMA vector, same length / shape as ``arr_in``
    """
    n = arr_in.shape[0]
    ewma_arr = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma_arr[0] = ewma_old
    for i in range(1, n):
        w += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / w

    return ewma_arr
