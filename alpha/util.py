import re

def load_fasta(filename):
    """
    return a diction of RNA
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


def load_fasta_with_ids(fasta_filename, ids_filename):
    """
    construct a dataset from a fasta file with IDs in  IDs_filename
    :param fasta_filename: the location of the fasta
    :param ids_filename: the location of the IDS
    :return: a dictionary with {ID: seq}
    """

    ret = {}

    with open(ids_filename, "r") as f:
        matcher = re.compile(">([^>\s|]*)[\s|]*")
        ids = [matcher.findall(l)[0] for l in f if len(matcher.findall(l)[0]) > 0]
        ids = set(ids)

    with open(fasta_filename, "r") as f:
        key = 'default_string'
        for l in f:
            l = l[:-1]
            if l[0] == '>' or l[0] == 'E' or l[0] == 'g':
                key = matcher.findall(l)[0]
                if key[0] == '>':
                    key = key[1:]
            else:
                if key in ids:
                    ret[key] = l
                key = 'default_string'

    return ret


def test1():
    load_fasta_with_ids(fasta_filename="/Users/xingyuwei/Downloads/gencode.v19.lncRNA_transcripts.fa",
                        ids_filename="/Users/xingyuwei/Downloads/rna_f/Human_lncRNA.txt")


if __name__ == '__main__':
    test1()
