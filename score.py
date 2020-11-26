import json
import numpy as np
import os
import pickle
import joblib
from azureml.core.model import Model
import azureml.train.automl


def init():
    global model

    model_path = Model.get_model_path(model_name='best_automl_model.pkl')

    model = joblib.load(model_path)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    y = model.predict(data)

    return y.tolist()
