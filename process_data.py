import os
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder


class ProcessData:
    def __init__(self, model_type: str, data_name:str, out_dir:str, test_size:float, target:str, cat_cols:list):
        self.model_type = model_type
        self.df = pd.read_csv(data_name)
        self.test_size = test_size
        self.target = target
        self.cat_cols = cat_cols
        self.out_dir = out_dir


    def clean_data(self):
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].astype('float64')
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', '0').astype('float64')
        self.df['Churn'] = self.df['Churn'].map({'Yes':1, 'No': 0})

    
    def split_data(self, df: pd.DataFrame, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
        
        if not val_size:
            val_size = test_size / (1 - test_size)

        train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

        self.X_train = train[train.columns.drop(target)]
        self.X_val = val[val.columns.drop(target)]
        self.X_test = test[test.columns.drop(target)]

        self.y_train = train[target]
        self.y_val = val[target]
        self.y_test = test[target]
        

    def lgb_data_processor(self, cat_cols):
        encoder = OneHotEncoder(
        variables=cat_cols,
        drop_last_binary=True
        )

        self.X_train = encoder.fit_transform(self.X_train)
        self.X_valid = encoder.transform(self.X_valid)
        self.X_test = encoder.transform(self.X_test)


    def cat_data_processor(self):
        pass

    
    def save_data(self, out_dir):
        for d in ('X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test'):
            self.__getattribute__(d).to_csv(os.path.join(out_dir, f'{d}.csv'), ignore_index=False)


    def run_pipeline(self):
        self.clean_data()
        self.split_data(df=self.df, test_size=self.test_size, target=self.target)
        self.__getattribute__(f'{self.model_type}_data_processor', cat_cols=self.cat_cols)
        self.save_data(out_dir=self.out_dir)