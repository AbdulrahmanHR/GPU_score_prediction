# inference
from data_preparation import DataPreparation
from models import HybridModels

class InferencePipeline:
    def __init__(self, model, data_file):
        self.model = model
        self.data_file = data_file
        self.data_prep = DataPreparation(self.data_file)

    def preprocess_new_data(self):
        return self.data_prep.preprocess_data()

    def predict(self, new_data):
        if self.model == 'xgboost_lstm':
            hybrid_model = HybridModels(new_data) # how to enter new data ?
            return hybrid_model.xgboost_lstm()
        elif self.model == 'lightgbm_lstm':
            hybrid_model = HybridModels(new_data)
            return hybrid_model.lightgbm_lstm()
        elif self.model == 'xgboost_cnn':
            hybrid_model = HybridModels(new_data)
            return hybrid_model.xgboost_cnn()
        elif self.model == 'lightgbm_cnn':
            hybrid_model = HybridModels(new_data)
            return hybrid_model.lightgbm_cnn()
        else:
            raise ValueError("Invalid model name. Choose from 'xgboost_lstm', 'lightgbm_lstm', 'xgboost_cnn', 'lightgbm_cnn'.")