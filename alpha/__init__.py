import numpy as np

from alpha import condonBias
from alpha.condonBias import getCodonFreq
from alpha.constants import kozak_dict, trna
from alpha.kmotif import getKmotif
from alpha.orf_finder import get_max_orf
from alpha.ribosome_coverage import get_ribosome_coverage
from alpha.sScore import findMaxInterval, sscores
from alpha.translate import hydrophobicity, complexity, get_amio_acid_comp
from alpha.zcurve import getZcurve


class feature_extractor():

    def __init__(self, sMatrix):
        self.sMatrix = sMatrix
        self.seq_cnt = 0

    def extact_features(self, seq):
        """
        Get features for sequence
        :param seq: the sequence of RNA transcript
        :return: a list of features
        """

        orf_length = 0
        orf_start = 0

        orf_start, orf_length = orf_finder.get_max_orf(seq)
        orf_coverage = (orf_length + 0.0) / (len(seq) + 0.0)

        kozak1, kozak2 = kmotif.getKmotif(seq, orf_start)
        kozak1b = np.zeros(25)
        kozak2b = np.zeros(25)
        kozak1b[kozak_dict[kozak1]] = 1
        kozak2b[kozak_dict[kozak2]] = 1

        zcurvel = zcurve.getZcurve(seq, orf_start, orf_length)
        sScorel = sScore.findMaxInterval(sScore.sscores(seq, self.sMatrix))
        condonBias_orf = condonBias.getCodonFreq(seq[orf_start:(orf_start + orf_length)])
        condonBias_t = condonBias.getCodonFreq(seq)

        # rc = get_ribosome_coverage(seq, orf_start, orf_length, trna)
        hp = hydrophobicity(seq[orf_start:(orf_start + orf_length)])
        dist = get_amio_acid_comp(seq[orf_start:(orf_start + orf_length)])
        cmp = complexity(dist)

        return [len(seq), orf_length, orf_coverage] + list(kozak1b) + list(kozak2b) + zcurvel \
               + list(sScorel) + list(condonBias_orf) + list(condonBias_t)  # + list(hp) + list(dist)


    def extract_features_using_dict(self, code, seq):
        """
    
        :param seq: the rna test sequence
        :param code: the ID of transcripts
        :param verbose: if verbose is 1, a serial number will be outputted.
        :return: a json with {name: feature}
        """
        features_dict = {'ID': code, 'seq': seq}

        try:
            orf_length = 0
            orf_start = 0

            orf_start, orf_length = get_max_orf(seq)
            orf_coverage = (orf_length + 0.0) / (len(seq) + 0.0)
            features_dict['orf_start'] = orf_start
            features_dict['orf_length'] = orf_length
            features_dict['orf_coverage'] = orf_coverage

            kozak1, kozak2 = getKmotif(seq, orf_start)
            kozak1b = np.zeros(25)
            kozak2b = np.zeros(25)
            kozak1b[kozak_dict[kozak1]] = 1
            kozak2b[kozak_dict[kozak2]] = 1
            features_dict['kozak1'] = kozak1
            features_dict['kozak2'] = kozak2

            for i in range(len(kozak1b)):
                features_dict['kozak1b_' + str(i)] = kozak1b[i]
            for i in range(len(kozak2b)):
                features_dict['kozak2b_' + str(i)] = kozak2b[i]

            zcurvel = getZcurve(seq, orf_start, orf_length)
            for i in range(len(zcurvel)):
                features_dict['zcurve_' + str(i)] = zcurvel[i]

            sScorel = findMaxInterval(sscores(seq, self.sMatrix))
            for i in range(len(sScorel)):
                features_dict['sScorel_' + str(i)] = sScorel[i]

            condonBias_orf = getCodonFreq(seq[orf_start:(orf_start + orf_length)])
            for i in range(len(condonBias_orf)):
                features_dict['condonBias_orf_' + str(i)] = condonBias_orf[i]

            condonBias_t = getCodonFreq(seq)
            for i in range(len(condonBias_t)):
                features_dict['condonBias_t_' + str(i)] = condonBias_t[i]

            # rc = get_ribosome_coverage(seq, orf_start, orf_length, trna)
            hp = hydrophobicity(seq[orf_start:(orf_start + orf_length)])
            dist = get_amio_acid_comp(seq[orf_start:(orf_start + orf_length)])
            cmp = complexity(dist)
            features_dict['hp'] = hp
            features_dict['cmpl'] = cmp

            return features_dict
            # return [len(seq), orf_length, orf_coverage] + list(kozak1b) + list(kozak2b) + zcurvel \
            #        + list(sScorel) + list(condonBias_orf) + list(condonBias_t)
        except:
            features_dict['exception'] = 'Yes'
            return features_dict


