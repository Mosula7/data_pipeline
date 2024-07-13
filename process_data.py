import os
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder


class ProcessData:
    def __init__(self, model_type: str, data_name: str, out_dir: str,
                 test_size: float, target: str,
                 cat_cols: list, binary_cols: list
                 ):

        self.model_type = model_type
        self.df = pd.read_csv(os.path.join('data', data_name))
        self.test_size = test_size
        self.target = target
        self.cat_cols = cat_cols
        self.binary_cols = binary_cols
        self.out_dir = out_dir

    def common_step(self):

        self.df = self.df.drop(columns=['customerID'])
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].astype('float64')
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', '0')
        self.df['TotalCharges'] = self.df['TotalCharges'].astype('float64')
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})

    def split_data(self, df: pd.DataFrame, target: str, test_size: float,
                   val_size: float = None, random_state: int = 0):

        if not val_size:
            val_size = test_size / (1 - test_size)

        train_valid, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df[target],
            random_state=random_state
        )
        train, valid = train_test_split(
            train_valid,
            test_size=val_size,
            stratify=train_valid[target],
            random_state=random_state
        )

        self.X_train = train[train.columns.drop(target)]
        self.X_valid = valid[valid.columns.drop(target)]
        self.X_test = test[test.columns.drop(target)]

        self.y_train = train[target]
        self.y_valid = valid[target]
        self.y_test = test[target]

    def lgb_data_processor(self):

        encoder = OneHotEncoder(
            variables=self.cat_cols,
            drop_last_binary=True
        )

        self.X_train = encoder.fit_transform(self.X_train)
        self.X_valid = encoder.transform(self.X_valid)
        self.X_test = encoder.transform(self.X_test)

    def cat_data_processor(self):

        encoder = OneHotEncoder(
            variables=self.binary_cols,
            drop_last_binary=True
        )

        self.X_train = encoder.fit_transform(self.X_train)
        self.X_valid = encoder.transform(self.X_valid)
        self.X_test = encoder.transform(self.X_test)

    def save_data(self, out_dir):

        datasets = (
            'X_train', 'X_valid', 'X_test',
            'y_train', 'y_valid', 'y_test'
        )
        for d in datasets:
            self.__getattribute__(d).to_csv(
                os.path.join(out_dir, f'{d}.csv'),
                index=False
            )

    def run_pipeline(self):

        self.common_step()
        self.split_data(
            df=self.df,
            test_size=self.test_size,
            target=self.target
        )
        self.__getattribute__(f'{self.model_type}_data_processor')()
        self.save_data(out_dir=self.out_dir)


if __name__ == '__main__':
    model_type = sys.argv[1]
    with open(os.path.join('configs', f'config_{model_type}.json')) as file:
        config = json.load(file)['data']

    data_pipeline = ProcessData(
            model_type=model_type,
            data_name=config["data_name"],
            out_dir=config["out_dir"],
            test_size=config["test_size"],
            target=config["target"],
            cat_cols=config["cat_cols"],
            binary_cols=config["binary_cols"]
        )

    data_pipeline.run_pipeline()
