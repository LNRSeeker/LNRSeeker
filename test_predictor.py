import logging

from predictor import predictor

logging.basicConfig(filename="something.log")
logger = logging.getLogger("temp")

if __name__ == '__main__':
    LNCRNA_TRAIN = "/Users/xingyuwei/NOTE/M1/lncrna_train_mini.fa"
    GENE_TRAIN = "/Users/xingyuwei/NOTE/M1/gene_train_mini.fa"

    model = predictor()
    model.logger = logger
    model.train(GENE_TRAIN, LNCRNA_TRAIN)

    import pickle as pkl
    with open("mymodel.pkl", "wb") as f:
        pkl.dump((model.atrans, model.fe, model.model), f)