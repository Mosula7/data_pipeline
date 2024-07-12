import numpy as np
import pandas as pd

class ProcessData:
    def __init__(self, model_type):
        self.model_type = model_type
        self.__getattribute__(f'{self.model_type}_data_processor')
        

    def lgb_data_processor():
        pass


    def cat_data_processor():
        pass