import os
import sys
import json

import pandas as pd
import catboost as cat
import lightgbm as lgb


class TrainModel:
    def __init__(self, model_type, data_dir, hyperparams):
        self.model_type = model_type
        self.data_dir = data_dir
        self.hyperparams = self.hyperparams
        datasetsets = (
            'X_train', 'X_valid', 'X_test',
            'y_train', 'y_valid', 'y_test'
        )
        for d in datasetsets:
            self.__setattr__(
                d,
                pd.read_csv(os.path.join(self.data_dir, f'{d}.csv'))
            )

    def train_lgb(self):
        model = lgb.LGBMClassifier(**self.hyperparams)
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_valid, self.y_valid)
        )

    def train_cat(self):
        model = cat.CatBoostClassifier(**self.hyperparams)
        model.fit()

    def evaluate_model(self):
        pass

    def run_pipeline(self):
        self.__getattribute__(f'train_{self.model_type}')
        self.evaluate_model()


if __name__ == '__main__':
    model_type = sys.argv[1]

    with open(os.path.join('configs', f'config_{model_type}.json')) as file:
        data_dir = json.load(file)['data']['out_dir']

    with open(os.path.join('configs', f'config_{model_type}.json')) as file:
        hyperparams = json.load(file)['model']

    model_pipeline = TrainModel(
        model_type=model_type,
        data_dir=data_dir,
        hyperparams=hyperparams
    )

    model_pipeline.run_pipeline()
