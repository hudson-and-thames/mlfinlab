import numpy as np

def _get_ascii_table():
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table

def quantile_mapping(array, num_letters=26):
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[num_letters]
    for q, l in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, q)] = l
    return encoding_dict

def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def enconde_array(array, encoding_dict):
    message = ''
    for el in array:
        message += get_letter_from_encoding(array, inverse_encoding_dict)
    return message

def get_letter_from_encoding(value, encoding_dict):
    return inverse_encoding_dict[_find_nearest(list(inverse_encoding_dict.keys()), value)]
