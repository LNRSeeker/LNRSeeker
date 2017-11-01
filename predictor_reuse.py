from  predictor import predictor
def reload_predictor(atrans, fe, model):
    ret = predictor()
    ret.model = model
    ret.atrans = atrans
    ret.fe = fe

    return fe