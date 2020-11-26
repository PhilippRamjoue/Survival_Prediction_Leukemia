import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model

    model_path = Model.get_model_path(model_name='AutoML37a31f86512')
    model = joblib.load(model_path)

    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    #model = Model.get_model_path('best_automl_model')

    #model = joblib.load(model_path)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    y = model.predict(data)

    return y.tolist()
