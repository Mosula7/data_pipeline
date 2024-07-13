import catboost as cat
import lightgbm as lgb

class TrainModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.__getattribute__(f'train_{self.model_type}')
    

    def train_lgb():
        pass


    def train_cat():
        pass