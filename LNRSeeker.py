import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from predictor import predictor
import pickle as pkl
from keras.models import Model, model_from_json

def check_args(args):

    home_prefix = os.getenv("HOME")
    model_file = args.model_file.replace("~", home_prefix)
    input_file = args.input_file.replace("~", home_prefix)

    try:
        f = open(model_file, "rb")
        f.close()
    except FileNotFoundError:
        logging.error("Cannot found the model")
        return False

    try:
        f = open(input_file, "r")
        f.close()
    except FileNotFoundError:
        logging.error("Cannot found file of transcripts")
        return False

    return True


def predict_one(seeker, seq):
    """
    give the prediction for a sequence.
    """
    data = seeker.fe.extract_features_using_dict("tempcode", seq)
    if 'exception' in data.keys():
        return 1
    df = pd.Series(data).drop(['ID', 'seq', 'kozak1', 'kozak2'])
    x = np.asarray(df)
    x = seeker.atrans.transform(x)
    x.shape = (1, 260)
    y_hat = seeker.model.predict(x)
    return y_hat[0]

def test(args):

    home_prefix = os.getenv("HOME")
    model_file = args.model_file.replace("~", home_prefix)
    input_file = args.input_file.replace("~", home_prefix)
    output_prefix = args.output_prefix

    with open(model_file, "rb") as f:
        configs = pkl.load(f)

    seeker = predictor()
    seeker.fe = configs["fe"]
    seeker.atrans = configs["atrans"]
    seeker.model = model_from_json(configs["config"])
    seeker.model.set_weights(configs["weights"])

    with open(input_file, "r") as f:
        lines = f.readlines()

    seqs = [lines[i+1][:-1] for i in range(0, len(lines), 2)]
    descs = [lines[i][:-1] for i in range(0, len(lines), 2)]
    probs = [predict_one(seeker, s) for s in seqs]

    with open(output_prefix + ".output", "w") as f:
        f.write("Probability\tDescription\n")
        for i in range(len(descs)):
            f.write("\t".join([str(float(probs[i])), descs[i]]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train LNRSeeker with transcripts database.")
    parser.add_argument(
        '-m', dest='model_file', type=str,
        help='the configuration outputted by LNRSeeker-train',
        required=True
    )
    parser.add_argument(
        '-i', dest='input_file', type=str,
        help='the fasta file of input',
        required=True
    )
    parser.add_argument(
        '-o', dest='output_prefix', type=str,
        help='the prefix for output file',
        required=True
    )

    args = parser.parse_args()
    print(args)

    if not check_args(args):
        sys.exit(1)

    test(args)

if __name__ == '__main__':
    main()
