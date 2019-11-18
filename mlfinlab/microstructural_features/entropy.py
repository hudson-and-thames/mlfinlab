"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In)
"""

import math
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
    Compute probability mass fucntion for a one-dim discete rv, Snippet 18.1, page 266.

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
    Get Plug-in entropy estimator, Snippet 18.1, page 266.

    :param message: (str or array) encoded message
    :param word_length: (int) approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 2
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out
