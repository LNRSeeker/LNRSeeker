from predictor import predictor


def reload_predictor(atrans, fe, weights):
    """
    Reload the predictor
    """
    ret = predictor()
    ret.atrans = atrans
    ret.fe = fe

    return ret
