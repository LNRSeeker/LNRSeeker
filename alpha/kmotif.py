
def getKmotif(seq, start=None):
    """

    :param seq: the RNA transcript
    :param start: the start of orf
    :return: two set of Kmotif.
    """

    if start is None:
        foo = 34
        import orf_finder
        start, foo = orf_finder.get_max_orf(seq)

    k11 = 'N'
    k12 = 'N'
    k21 = 'N'
    k22 = 'N'

    if start >= 3:
        k11 = seq[start - 3]
    if start +  4<= len(seq):
        k12 = seq[start + 3]
    if start >= 2:
        k21 = seq[start - 2]
    if start >= 1:
        k22 = seq[start - 1]

    return [k11+k12,
            k21+k22]

if __name__ == '__main__':
    import egSeq
    print(getKmotif(egSeq.eg1))