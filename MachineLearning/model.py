import time

from joblib import dump, load
import pandas as pd
from pandas.core.frame import DataFrame
from MachineLearning.data import data_class
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class model:
    save_folder = 'MachineLearning/models'
    model = None
    today = int(time.time())

    def __init__(self, labels: list) -> None:
        self.labels = labels

    def __preprocess(self, data: data_class, train:bool):
        df = data.filter_features(base=False, feature=True, ratio=True)
        # TODO: x, y
        x = ''
        y = ''
        # TODO: train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)
        if train:
            train_x, train_y, test_x, test_y = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)
            output = (train_x, train_y, test_x, test_y)
        else:
            output = (x, y)
        return df, output
    
    def load(self) -> None:
        #TODO: load model from file, if fail raise error
        #TODO: load other objects we might need (scaler, normalizer, pca??)
        return

    def create_model(self):
        self.model = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
        return self.model
    
    def train(self, data: data_class) -> None:
        df, (x, y) = self.__preprocess(self, data, True)

        # create model if there is none
        if self.model == None:
            self.create_model()

        self.model.fit(x,y)
        pass

    def predict(self, data: data_class) -> list:
        df = self.__preprocess(self, data)
        if self.model == None:
            self.load()
        prediction = self.model.predict(df)
        return prediction
