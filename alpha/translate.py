import math

import numpy as np

trans = "FFLLLLLLIIIMVVVVSSSSPPPPTTTTAAAAYY**HHQQNNKKDDEECC*WRRRRSSRRGGGG"

this_four = ["T", "C", "A", "G"]

tri = [[[x + y + z for z in this_four]
        for x in this_four]
       for y in this_four]

tri_d = np.ndarray.flatten(np.asarray(tri))

hyd = {
    'A': 0.61,
    'R': 0.60,
    'N': 0.06,
    'D': 0.46,
    'C': 1.07,
    'Q': 0,
    'E': 0.47,
    'G': 0.07,
    'H': 0.61,
    'I': 2.22,
    'L': 1.53,
    'K': 1.15,
    'M': 1.18,
    'F': 2.02,
    'P': 1.95,
    'S': 0.05,
    "T": 0.05,
    "W": 2.65,
    "Y": 1.88,
    "V": 1.32,
    '*': 0
}

# mouse


# human_protein_distribution = [
#     1.81,
#     5.14,
#     6.10,
#     3.83,
#     6.91,
#     2.17,
#     5.48,
#     6.01,
#     9.16,
#     2.50,
#     4.33,
#     4.95,
#     3.98,
#     5.86,
#     6.84,
#     5.77,
#     6.65,
#     1.34,
#     3.15
# ]

human_protein_distribution = [
    8.76,
    1.38,
    5.49,
    6.32,
    3.87,
    7.03,
    2.26,
    5.49,
    5.19,
    9.68,
    2.32,
    3.93,
    5.02,
    3.9,
    5.78,
    7.14,
    5.53,
    6.73,
    1.25,
    2.91
]
mouse_protein_distribution = [
    6.8426,
    2.3778,
    4.7156,
    6.662,
    3.9129,
    6.4814,
    2.5986,
    4.545,
    5.5684,
    10.1635,
    2.2875,
    3.6019,
    6.0299,
    4.6253,
    5.5383,
    8.2974,
    5.448,
    6.2105,
    1.2541,
    2.8394
]

protein_distribution = human_protein_distribution


def getIndex(s):
    if s == 'T':
        return 0
    if s == 'C':
        return 1
    if s == 'A':
        return 2
    else:
        return 3

def getIndex3(s):
    ss = [getIndex(x) for x in s]
    return ss[0] + ss[1]*4 + ss[2] * 16

l = "ACDEFGHIKLMNPQRSTVWY"
pro_dict = {x: l.index(x) for x in l}


def get_amio_acid_comp(orf):
    """
    :param orf: Open Reading Frame
    :return: a list of
    """
    if len(orf) == 0:
        return [0 for x in range(20)]
    ret = [0.0 for i in range(20)]
    for i in range(0, len(orf), 3):
        if 'N' not in orf[i:i + 3]:
            if trans[getIndex3(orf[i:i + 3])] == '*':
                continue
            ret[pro_dict[trans[getIndex3(orf[i:i + 3])]]] += 1.0

    return [x / (len(orf) + 0.0) for x in ret]


def hydrophobicity(orf):
    ret = 0
    for i in range(0, len(orf), 3):
        if i + 3 <= len(orf) and 'N' not in orf[i:i + 3]:
            ret += hyd[trans[getIndex3(orf[i:i+3])]]

    return ret


def entropy(p, q):
    if p == 0:
        return 0.0
    else:
        return p * math.log(p/q)


def complexity(distribution=human_protein_distribution):
    return sum([entropy(distribution[i],
                        protein_distribution[i]/100)
                for i in range(len(distribution))])
