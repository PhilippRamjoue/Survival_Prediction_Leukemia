import json
import numpy as np
import os
import pickle
#import joblib
from sklearn.external import joblib
from azureml.core.model import Model
import azureml.train.automl


def init():
    global model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    #model = Model.get_model_path('best_automl_model')

    model = joblib.load(model_path)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    y = model.predict(data)

    return y.tolist()
