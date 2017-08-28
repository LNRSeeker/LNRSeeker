"""
util.py
useful functions
"""


def load_fasta(filename):
    """
    resolve the fasta file as a map in the form of  {id:seq}
    """
    ret = {}
    with open(filename, 'r') as f:
        k = 'k'
        for line in f:
            line = line[:-1]
            if line[0] == '>' or line[0] == 'E':
                k = line
            else:
                ret[k] = line

    return ret

