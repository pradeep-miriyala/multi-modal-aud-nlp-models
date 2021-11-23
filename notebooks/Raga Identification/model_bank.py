import numpy as np
import pandas as pd
import torch


def load_model(mdl_file, model_class, **model_args):
    m = model_class(**model_args)
    m.load_state_dict(torch.load(mdl_file))
    return m


def predict_probabilities(mdl, spec):
    if spec.shape[1] < 9601:
        spec = torch.hstack((spec, torch.zeros(40, 9601 - spec.shape[1])))
    spec = spec.reshape((1, 1, 40, 9601))
    mdl.eval()
    with torch.no_grad():
        r = softmax(mdl(spec).detach().numpy())
    return [r[0][0], r[0][1]]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def load_models(models, model_class, **model_args):
    return {k: load_model(models[k], model_class, **model_args) for k in models}


def predict(model_banks, spec):
    models = {k: predict_probabilities(model_banks[k], spec) for k in model_banks}
    predictions = pd.DataFrame(models).transpose().sort_values(by=1, ascending=False)
    return predictions


def get_raga_name(predictions):
    return predictions.head(1)[1].index[0], predictions.head(1)[1][predictions.head(1)[1].index[0]]
