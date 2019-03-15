import numpy as np
from numba import jit
from numba import float64
from numba import int64


@jit((float64[:], int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    """Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'
    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``
    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma[i] = ewma_old / w

    return ewma
