import time

from joblib import dump, load

from MachineLearning.data import data_class


class model:
    save_folder = 'MachineLearning/models'
    model = None
    today = int(time.time())

    def __init__(self, labels: list) -> None:
        self.labels = labels

    def __preprocess(self):
        return
    
    def load(self):
        return

    def train(self):
        pass

    def predict(self, data: data_class) -> list:
        self.df = data.filter_features(base=False, feature=True, ratio=True)
        self.__preprocess()
        return
