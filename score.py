import json
import numpy as np
import pandas as pd
import os
import pickle
import joblib

def init():
    global model

    model_path =  os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')

    model = joblib.load(model_path)

def run(raw_data):

    #data = pd.read_json(raw_data,orient='records')
    #y = model.predict(data)

    data = pd.DataFrame(json.loads(raw_data)['data'])

    y = model.predict(data)

    return y.tolist()
