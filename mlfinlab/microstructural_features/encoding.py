"""
Various functions for message encoding (quantile)
"""
import numpy as np


def encode_tick_rule_array(tick_rule_array: list) -> str:
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) of tick rules
    :return: (str) encoded message
    """
    message = ''
    for element in tick_rule_array:
        if element == 1:
            message += 'a'
        elif element == -1:
            message += 'b'
        elif element == 0:
            message += 'c'
        else:
            raise ValueError('Unknown value for tick rule: {}'.format(element))
    return message


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) of ASCII symbols
    """
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) of values to split on quantiles
    :param num_letters: (int) number of letters(quantiles) to encode
    :return: (dict) of quantile-symbol
    """
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for quant, letter in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, quant)] = letter
    return encoding_dict


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) of values to split on quantiles
    :param step: (float) discretization step (sigma)
    :return: (dict) of value-symbol
    """
    i = 0
    ascii_table = _get_ascii_table()
    encoding_dict = {}
    encoding_steps = np.arange(min(array), max(array), step)
    for element in encoding_steps:
        try:
            encoding_dict[element] = ascii_table[i]
        except IndexError:
            raise ValueError(
                'Length of dictionary ceil((max(arr) - min(arr)) / step = {} is more than ASCII table lenght)'.format(
                    len(encoding_steps)))
        i += 1
    return encoding_dict


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) of values
    :param value: (float) value for which the nearest element needs to be found
    :return: (float) the nearest to the value element in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int)
    :param encoding_dict: (dict)
    :return: (str): letter from encoding dict
    """
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]


def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) of values to encode
    :param encoding_dict: (dict) of quantile-symbol
    :return: (str): encoded message
    """
    message = ''
    for element in array:
        message += _get_letter_from_encoding(element, encoding_dict)
    return message
