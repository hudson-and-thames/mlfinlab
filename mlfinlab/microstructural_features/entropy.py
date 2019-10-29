def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_letter_from_encoding(value, inverse_encoding_dict):
    return inverse_encoding_dict[_find_nearest(list(inverse_encoding_dict.keys()), value)]

def get_shannon_entropy(message):
    """
    """
    entropy = 0
    # There are 256 possible ASCII characters
    for character_i in range(256):
        prob = message.count(chr(character_i)) / len(message)
        if prob > 0:
            entropy += - prob * math.log(prob, 2)
    return entropy

def lempel_ziv_entropy(message):
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message = message[i:j + 1]
            if message not in lib:
                lib.append(message)
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
