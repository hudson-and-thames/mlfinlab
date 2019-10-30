import numpy as np


def encode_tick_rule_array(tick_rule_array):
    message = ''
    for el in tick_rule_array:
        if el == 1:
            message += 'a'
        elif el == -1:
            message += 'b'
        elif el == 0:
            message += 'c'
        else:
            raise ValueError('Unknown value for tick rule: {}'.format(el))
    return message


def _get_ascii_table():
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table


def quantile_mapping(array, num_letters=26):
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for q, l in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, q)] = l
    return encoding_dict


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value, encoding_dict):
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]


def encode_array(array, encoding_dict):
    message = ''
    for el in array:
        message += _get_letter_from_encoding(el, encoding_dict)
    return message
