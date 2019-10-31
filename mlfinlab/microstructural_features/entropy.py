import math
import numpy as np


def get_shannon_entropy(message):
    """
    """
    log2 = lambda x: math.log(x) / math.log(2)
    exr = {}
    entropy = 0
    for each in message:
        try:
            exr[each] += 1
        except:
            exr[each] = 1
    textlen = len(message)
    for k, v in exr.items():
        freq = 1.0 * v / textlen
        entropy += freq * log2(freq)
    entropy *= -1
    return entropy


def get_lempel_ziv_entropy(message):
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message_ = message[i:j + 1]
            if message_ not in lib:
                lib.append(message_)
                break
        i = j + 1
    return len(lib) / len(message)


def get_plug_in_entropy(message, w=None):
    if w is None:
        w = 2

    def pmf1(message, w):
        lib = {}
        if not isinstance(message, str):
            message = ''.join(map(str, message))
        for i in range(w, len(message)):
            message_ = message[i - w:i]
            if message_ not in lib:
                lib[message_] = [i - w]
            else:
                lib[message_] = lib[message_] + [i - w]
        pmf = float(len(message) - w)
        pmf = {i: len(lib[i]) / pmf for i in lib}
        return pmf

    pmf = pmf1(message, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out
