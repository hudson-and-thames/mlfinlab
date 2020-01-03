"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from typing import Union

import numpy as np


def get_shannon_entropy(message: str) -> float:
    """
    Get Shannon entropy from message, page 263-264.

    :param message: (str) encoded message
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
    Get Lempel-Ziv entropy estimate, Snippet 18.2, page 266.

    :param message: (str) encoded message
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
    Compute probability mass function for a one-dim discete rv, Snippet 18.1, page 266.

    :param message: (str or array) encoded message
    :param word_length: (int) approximate word length
    :return: (dict) of pmf for each word from message
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
    Get Plug-in entropy estimator, Snippet 18.1, page 265.

    :param message: (str or array) encoded message
    :param word_length: (int) approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 1
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out


def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:
    """
    Snippet 18.3, Function That Computes the Length of the Longest Match, p.267
    :param message: (str or array) encoded message
    :start_index: (int) start index for search
    :window: (int) window length
    :return: (int, str) match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = np.empty(shape=0)
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
    Get Kontoyiannis entropy
    Snippet 18.4, Implementations of Algorithms Discussed in Gao et al.[2008]
    :param message: (str or array) encoded message
    :param window: (int) expanding window length, can be negative
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
    out['r'] = 1 - out['h'] / np.log2(len(message))  # Redundancy, 0<=r<=1
    return out['h']
