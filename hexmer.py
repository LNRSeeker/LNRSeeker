"""
hexmer.py
Find the S-Score for each hexmer
"""

import numpy as np
from util import load_fasta
import argparse
import pickle as pkl
import os, sys

# ANTLookupTable
four = ['A', 'C', 'G', 'T']
ANTLoookupTable = [[[i + j + k for i in four]
                    for j in four] for k in four]
ANTLookupTableFlat = np.ndarray.flatten(np.asarray(ANTLoookupTable))
ANTLookupTableDict = {ANTLookupTableFlat[i]: i
                      for i in range(len(ANTLookupTableFlat))}

def get_score_matrix(GENE_FILE, LNCRNA_FILE):
    """
    get S-score matrix for each TNP
    :param GENE_FILE: the filename of the fasta file for coding transcript.
    :param LNCRNA_FILE: the filename of the fasta file for noncoding transcript.
    :return: a 64x64 matrix indicating the s-score for each 
    """
    geneSeqs = load_fasta(GENE_FILE)
    geneM = np.zeros(shape=(64, 64))
    geneTotal = 0
    for k, seq in geneSeqs.items():
        seqM = np.zeros(shape=(64, 64))
        for i in range(len(seq) - 5):
            a = seq[i:i + 3]
            b = seq[i + 3:i + 6]
            if (not a in ANTLookupTableDict.keys()) or (not b in ANTLookupTableDict.keys()):
                continue
            seqM[ANTLookupTableDict[a]][ANTLookupTableDict[b]] += 1
        geneM = geneM + seqM
        if len(seq) - 6 >= 0:
            geneTotal = geneTotal + len(seq) - 5

    geneM /= geneTotal

    # In[47]:

    # 统计ANT在lncRNA中的出现频率
    lncrnaSeqs = load_fasta(LNCRNA_FILE)
    lncrnaM = np.zeros(shape=(64, 64))
    lncrnaTotal = 0
    for k, seq in lncrnaSeqs.items():
        seqM = np.zeros(shape=(64, 64))
        for i in range(len(seq) - 5):
            a = seq[i:i + 3]
            b = seq[i + 3:i + 6]
            if (not a in ANTLookupTableDict.keys()) or (not b in ANTLookupTableDict.keys()):
                continue
            seqM[ANTLookupTableDict[a]][ANTLookupTableDict[b]] += 1
        lncrnaM = lncrnaM + seqM
        if len(seq) - 6 >= 0:
            lncrnaTotal = lncrnaTotal + len(seq) - 5

    lncrnaM /= lncrnaTotal

    # 给出S-score需要的数值
    usageFreq = np.log(geneM / lncrnaM)

    return usageFreq


def main(gene_file="", lncrna_file="", output_prefix="human"):

    matrix = get_score_matrix(gene_file, lncrna_file)

    with open(output_prefix + ".pkl", "wb") as f:
        pkl.dump(matrix, f)

if __name__ == '__main__':

    # parse the arguments
    parser = argparse.ArgumentParser(description="Get the hexmer S-score")
    parser.add_argument(
        '-c', dest='coding_file',
        help='the fasta file of coding transcripts'
    )
    parser.add_argument(
        '-n', dest='non_coding_file',
        help='the fasta file of noncoding transcripts'
    )
    parser.add_argument(
        '-o', dest='output_prefix',
        help='the output file for the S-score matrix'
    )

    args = parser.parse_args()
    main(args.coding_file, args.non_coding_file, args.output_prefix)
