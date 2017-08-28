from alpha.orf_finder import get_max_orf


def getZcurve(seq, orfStart=None, orfLength=None):
    if orfStart is None:

        orfStart, orfLength = get_max_orf(seq)

    orf = seq[orfStart:(orfStart + orfLength)]

    First = [orf[i] for i in range(len(orf)) if i % 3 == 0]
    Second = [orf[i] for i in range(len(orf)) if i % 3 == 1]
    Third = [orf[i] for i in range(len(orf)) if i % 3 == 2]

    s = [First, Second, Third]
    ref = ['A', 'T', 'G', 'C']

    cnt = [[section.count(c) / (len(section) + 0.0) for c in ref]
           for section in s]
    u = [[(a[0] + a[2]) - (a[1] + a[3]),
          (a[0] + a[1]) - (a[2] + a[3]),
          (a[0] + a[3]) - (a[2] + a[1])] for a in cnt]
    [x, y, z] = [[l[i] for l in u] for i in range(3)]
    x = [a - sum(x) / 3.0 for a in x]
    y = [a - sum(y) / 3.0 for a in y]
    z = [a - sum(z) / 3.0 for a in z]

    First_pair = [First[i] + Second[i] for i in range(len(First))]
    Second_pair = [Second[i] + Third[i] for i in range(len(Second))]

    freq1 = [[First_pair.count(key1 + key2) / (len(First_pair) + 0.0) \
              for key2 in ref] for key1 in ref]
    freq2 = [[Second_pair.count(key1 + key2) / (len(Second_pair) + 0.0) \
              for key2 in ref] for key1 in ref]

    u12 = [[(a[0] + a[2]) - (a[1] + a[3]),
            (a[0] + a[1]) - (a[2] + a[3]),
            (a[0] + a[3]) - (a[2] + a[1])] for a in freq1]
    u23 = [[(a[0] + a[2]) - (a[1] + a[3]),
            (a[0] + a[1]) - (a[2] + a[3]),
            (a[0] + a[3]) - (a[2] + a[1])] for a in freq2]

    return x + y + z + u12[0] + u12[1] + u12[2] + u12[3] + u23[0] + u23[1] + u23[2] + u23[3]

if __name__ == '__main__':
    import egSeq
    print(getZcurve(egSeq.eg1))

