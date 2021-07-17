import tensorflow.keras as keras
import numpy as np 

MODEL_PATH = 'model.h5'

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        'down',
        'off',
        'on',
        'no',
        'yes',
        'stop',
        'up',
        'right',
        'left',
        'go'
    ]

    _instance = None 

    def predict(self, file_path):

        # extrat MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficients)
        
        # convert 2D MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make predictions
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.5, 0.3, ...] ]
        predicted_index = np.argmax(predictions)

    def preprocess(self, file_path):
        pass

def Keyword_spotting_Service():

    # ensure that we only have 1 instance
    if _Keyword_Spotting_Service._instance is None:
        Keyword_spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.model.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

    