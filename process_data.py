import pandas as pd
from sklearn.model_selection import train_test_split

class ProcessData:
    def __init__(self, model_type, df, test_size, target):
        self.model_type = model_type
        self.df = df
        self.test_size = test_size
        self.target = target

        self.clean_data()
        self.split_data(test_size=self.test_size, target=self.target)

        self.__getattribute__(f'{self.model_type}_data_processor')


    def clean_data(self):
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].astype('float64')
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', '0').astype('float64')
        self.df['Churn'] = self.df['Churn'].map({'Yes':1, 'No': 0})

    
    def split_data(self, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
        
        if not val_size:
            val_size = test_size / (1 - test_size)

        train_val, test = train_test_split(self.df, test_size=test_size, stratify=self.df[target], random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

        self.X_train = train[train.columns.drop(target)]
        self.X_val = val[val.columns.drop(target)]
        self.X_test = test[test.columns.drop(target)]

        self.y_train = train[target]
        self.y_val = val[target]
        self.y_test = test[target]
        

    def lgb_data_processor():
        pass


    def cat_data_processor():
        pass