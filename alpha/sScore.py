# coding=utf-8

import pickle
import profile

import numpy as np

from .util import load_fasta

# ANTLookupTable
LNCRNA_TRAIN_FA = '/Users/xingyuwei/Downloads/human_gencode_v17_lncRNA_22389'
GENE_TRAIN_FA = '/Users/xingyuwei/Downloads/human_rna_fna_refseq_mRNA_22389'
MATRIX_PKL = "/Users/xingyuwei/NOTE/data/human_sscoreMatrix.pkl"

# LNCRNA_TRAIN_FA = '/Users/xingyuwei/NOTE/other/Xenopus_tropicalis.JGI_4.2.72.ncrna.fa.1row.200_filt4'
# GENE_TRAIN_FA = '/Users/xingyuwei/NOTE/other/zebrafish.rna.fna.20130813.fa.1row.200_filt5'
# MATRIX_PKL = "/Users/xingyuwei/NOTE/other/zebrafish_sscoreMatrix.pkl"

if False:
    LNCRNA_TRAIN_FA = "/Users/xingyuwei/NOTE/other/Bos_taurus.UMD3.1.72.ncrna.fa.1row.200_filt4"
    GENE_TRAIN_FA = "/Users/xingyuwei/NOTE/other/cow.rna.fna.refseq.20130911.fa.1row.200_filt5"
    MATRIX_PKL = "/Users/xingyuwei/NOTE/other/cow_sscoreMatrix.pkl"

four = ['A', 'C', 'G', 'T']
ANTLoookupTable = [[[i + j + k for i in four]
                    for j in four] for k in four]
# In[31]:

ANTLookupTableFlat = np.ndarray.flatten(np.asarray(ANTLoookupTable))

print(ANTLookupTableFlat)
# In[32]:

ANTLookupTableDict = {ANTLookupTableFlat[i]: i
                      for i in range(len(ANTLookupTableFlat))}


def getIndex(s):
    if s == 'A':
        return 0
    if s == 'C':
        return 1
    if s == 'G':
        return 2
    else:
        return 3

def getIndex3(s):
    ss = [getIndex(x) for x in s]
    return ss[0] + ss[1]*4 + ss[2] * 16


# noinspection PyNonAsciiChar
def loadSScoreMatrix(filename=MATRIX_PKL):
    try:
        with open(filename, "rb") as f:
            usageFreq = pickle.load(f)
    except FileNotFoundError:
        # 统计ANT在coding RNA中的出现频率
        geneSeqs = load_fasta(GENE_TRAIN_FA)
        geneM = np.zeros(shape=(64, 64))
        geneTotal = 0
        for k, seq in geneSeqs.items():
            seqM = np.zeros(shape=(64, 64))
            for i in range(len(seq) - 5):
                a = seq[i:i + 3]
                b = seq[i + 3:i + 6]
                if a in ANTLookupTableDict.keys() and b in ANTLookupTableDict.keys():
                    seqM[ANTLookupTableDict[a]][ANTLookupTableDict[b]] += 1
            geneM += seqM
            if len(seq) - 6 >= 0:
                geneTotal = geneTotal + len(seq) - 5

        geneM /= geneTotal

        # In[47]:

        # 统计ANT在lncRNA中的出现频率
        lncrnaSeqs = load_fasta(LNCRNA_TRAIN_FA)
        lncrnaM = np.zeros(shape=(64, 64))
        lncrnaTotal = 0
        for k, seq in lncrnaSeqs.items():
            seqM = np.zeros(shape=(64, 64))
            for i in range(len(seq) - 5):
                a = seq[i:i + 3]
                b = seq[i + 3:i + 6]
                if a in ANTLookupTableDict.keys() and b in ANTLookupTableDict.keys():
                    seqM[ANTLookupTableDict[a]][ANTLookupTableDict[b]] += 1
            lncrnaM += seqM
            if len(seq) - 6 >= 0:
                lncrnaTotal = lncrnaTotal + len(seq) - 5

        lncrnaM /= lncrnaTotal

        # In[48]:

        # 给出S-score需要的数值
        usageFreq = np.log(geneM / lncrnaM)
        with open(filename, "wb") as pf:
            pickle.dump(usageFreq, pf)

    return usageFreq


def sscores(seq, usageFreq):
    """
    caculate S-scores in 6 reading frames
    """
    N = 150
    ret = [np.zeros((3, 3))]

    for off in [0, 1, 2]:
        seqi = seq[off:]
        sScoresEach = [usageFreq[getIndex3(seqi[i:i+3])][getIndex3(seqi[i+3:i+6])]
                       for i in range(len(seqi) - 5)]
        sScoresCumsum = np.ndarray.cumsum(np.asarray(sScoresEach))
        sScores = [sScoresCumsum[N]] + ([sScoresCumsum[N+i] - sScoresCumsum[i-i]
                                             for i in range(1, len(sScoresCumsum) - N)])
        ret.append(sScores)

    rev = list(seq[:])
    (rev.reverse())
    temp = ""
    for s in rev:
        temp = temp + s

    rev = temp[:]

    for off in [0, 1, 2]:
        seqi = seq[off:]
        sScoresEach = [usageFreq[getIndex3(seqi[i:i+3])][getIndex3(seqi[i+3:i+6])]
                       for i in range(len(seqi) - 5)]
        sScoresCumsum = np.ndarray.cumsum(np.asarray(sScoresEach))
        sScores = [sScoresCumsum[N]] + ([sScoresCumsum[N+i] - sScoresCumsum[i-i]
                                             for i in range(1, len(sScoresCumsum) - N)])
        ret.append(sScores)


    return ret[1:]

def findMaxInterval(scores):
    N = min(150, len(scores[0]))
    candidate = {}
    for i in range(6):
        curScores = scores[i]
        b = np.zeros(shape=(len(curScores)))
        m = np.zeros(shape=(len(curScores)))
        b[0] = curScores[0]
        m[0] = 0
        for i in range(1, len(curScores)):
            if b[i-1] > 0:
                b[i] = b[i-1] + curScores[i]
                m[i] = m[i-1]
            else:
                b[i] = curScores[i]
                m[i] = i

        maxi = 0
        for i in range(1, len(curScores)):
            if b[i] > b[maxi]:
                maxi = i
        candidate[b[maxi]] = [i, b[maxi] + i%3, maxi + 6 + i%3 ] # 注意是[i, j)
        # canSScores[off] = b[maxi]
        # canInter[off][0] = m[maxi] + off
        # canInter[off][1] = maxi + 6 + off

    maxS = max(candidate.keys())
    maxL = abs(candidate[maxS][2] - candidate[maxS][1])
    maxLP = maxL / sum([abs(candidate[s][2] - candidate[s][1]) for s in candidate.keys()])
    maxSD = sum([maxS - candidate[s][1] for s in candidate.keys()])
    return [maxS, maxL, maxLP, maxSD]


if __name__ == '__main__':
    profile.run("findMaxInterval(sscores(egSeq.eg1))")