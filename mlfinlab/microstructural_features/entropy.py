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

    pass


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """

    pass


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """

    pass


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """

    pass


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

    pass


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """

    pass
