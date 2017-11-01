import os
import sys
import argparse
import logging
from predictor import predictor
import pickle as pkl
from keras.models import Model, model_from_config

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
    seeker.model = model_from_config(configs["config"])
    seeker.model.set_weights(configs["weights"])

    with open(input_file, "r") as f:
        lines = f.readlines()

    seqs = [lines[i+1][:-1] for i in range(0, len(lines), 2)]
    descs = [lines[i][:-1] for i in range(0, len(lines), 2)]
    probs = [seeker.predict(s) for s in seqs]

    with open(output_prefix + ".output", "w") as f:
        f.write("Probability\tDescription\n")
        for i in range(len(descs)):
            f.write("%.8f\t%s\n".format(probs[i], descs[i]))
    

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
