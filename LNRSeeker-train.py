import argparse
import os, sys
import logging
import pickle as pkl

def train(args):

    from predictor import predictor

    logging.basicConfig(level=logging.INFO, filename="something.log")
    logger = logging.getLogger("temp")

    home_prefix = os.getenv("HOME")
    coding_file = args.coding_file.replace("~", home_prefix)
    non_coding_file = args.non_coding_file.replace("~", home_prefix)
    output_prefix = args.output_prefix

    seeker = predictor()
    seeker.logger = logger
    seeker.train(coding_file, non_coding_file)

    graph = {"atrans":seeker.atrans,
             "fe":seeker.fe,
             "config":seeker.model.to_json(),
             "weights":seeker.model.get_weights()}

    with open(output_prefix + "_config.pkl", "wb") as f:
        pkl.dump(graph, f)
        logger.info("The configuration has been dumped into " + output_prefix + "_config.pkl")

    # for debugging
    return graph


def check_args(args):

    home_prefix = os.getenv("HOME")
    coding_file = args.coding_file.replace("~", home_prefix)
    non_coding_file = args.non_coding_file.replace("~", home_prefix)

    try:
        f = open(coding_file, "r")
        f.close()
    except FileNotFoundError:
        logging.error("Cannot found file for coding transcripts")
        return False

    try:
        f = open(non_coding_file, "r")
        f.close()
    except FileNotFoundError:
        logging.error("Cannot found file for coding transcripts")
        return False

    return True

    

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train LNRSeeker with transcripts database.")
    parser.add_argument(
        '-c', dest='coding_file', type=str,
        help='the fasta file of coding transcripts',
        required=True
    )
    parser.add_argument(
        '-n', dest='non_coding_file', type=str,
        help='the fasta file of noncoding transcripts',
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

    train(args)

if __name__ == '__main__':
    main()

