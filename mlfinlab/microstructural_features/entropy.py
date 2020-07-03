"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from typing import Union

import numpy as np
from numba import njit


def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """
    exr = {}
    entropy = 0
    for each in message:
        try:
            exr[each] += 1
        except KeyError:
            exr[each] = 1
    textlen = len(message)
    for value in exr.values():
        freq = 1.0 * value / textlen
        entropy += freq * math.log(freq) / math.log(2)
    entropy *= -1
    return entropy


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message_ = message[i:j + 1]
            if message_ not in lib:
                lib.append(message_)
                break
        i = j + 1
    return len(lib) / len(message)


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    lib = {}
    if not isinstance(message, str):
        message = ''.join(map(str, message))
    for i in range(word_length, len(message)):
        message_ = message[i - word_length:i]
        if message_ not in lib:
            lib[message_] = [i - word_length]
        else:
            lib[message_] = lib[message_] + [i - word_length]
    pmf = float(len(message) - word_length)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 1
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out


@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = ''
    for length in range(window):
        msg1 = message[start_index: start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j: j + length + 1]
            if len(msg1) != len(msg0):
                continue
            if msg1 == msg0:
                sub_str = msg1
                break  # Search for higher l.
    return len(sub_str) + 1, sub_str  # Matched length + 1


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    out = {
        'h': 0,
        'r': 0,
        'num': 0,
        'sum': 0,
        'sub_str': []
    }
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = _match_length(message, i, i)
            out['sum'] += np.log2(i + 1) / length  # To avoid Doeblin condition
        else:
            length, msg_ = _match_length(message, i, window)
            out['sum'] += np.log2(window + 1) / length  # To avoid Doeblin condition
        out['sub_str'].append(msg_)
        out['num'] += 1
    try:
        out['h'] = out['sum'] / out['num']
    except ZeroDivisionError:
        out['h'] = 0
    out['r'] = 1 - out['h'] / (np.log2(len(message)) if np.log2(len(message)) > 0 else 1)  # Redundancy, 0<=r<=1
    return out['h']
