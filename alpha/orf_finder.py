"""Orf_finder
Finding the longest orf
Beware that we only search for the positive reading frame.
"""


def is_aug(s):
    return s == 'ATG'


def is_end(s):
    return s in ["TAG", "TAA", "TGA"]


def opp(c):
    if c == 'A':
        return 'T'
    elif c == 'T':
        return 'A'
    elif c == 'G':
        return 'C'
    else:
        return 'G'


def find_orf(seq):
    # pattern = "ATG(.{3})*(TAG|TAA|TGA)"
    flag = False
    curpos = 0
    curlen = 0
    maxpos = 0
    maxlen = 0
    for i in range(0, len(seq), 3):
        if i+3 <= len(seq):
            code = seq[i:i+3]
            if not flag and is_aug(code):
                flag = True
                curpos = i
                curlen = 3
            elif flag and is_end(code):
                flag = False
                curlen += 3
                if curlen > maxlen:
                    maxlen = curlen
                    maxpos = curpos
            elif flag:
                curlen += 3
    return [maxpos, maxlen]


def get_max_orf(seq):
    """
    Find the longest open reading frame and return
     its length and its start position.
     Notes that we only find the forwarding frames.
    :param seq: the transcript
    :return:
    """
    maxpos = 0
    maxlen = 0

    maxpos, maxlen = find_orf(seq)

    seq1 = seq[1:]
    temp_maxpos, temp_maxlen = find_orf(seq1)

    if temp_maxlen > maxlen:
        maxlen = temp_maxlen
        maxpos = temp_maxpos+1

    seq2 = seq[2:]
    temp_maxpos, temp_maxlen = find_orf(seq2)
    if temp_maxlen > maxlen:
        maxlen = temp_maxlen
        maxpos = temp_maxpos+2

    return [maxpos, maxlen]

if __name__ == '__main__':
    import egSeq
    eg1 = egSeq.eg1
    print(get_max_orf(eg1))