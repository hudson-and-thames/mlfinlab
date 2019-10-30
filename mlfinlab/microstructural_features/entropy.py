def get_shannon_entropy(message):
    """
    """
    entropy = 0
    for character_i in message:
        prob = message.count(character_i) / len(message)
        if prob > 0:
            entropy += - prob * math.log(prob, 2)
    return entropy

def get_lempel_ziv_entropy(message):
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
