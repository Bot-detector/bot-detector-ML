import logging
from math import log
import os
import time
import numpy as np

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from MachineLearning.data import data_class


class model:
    save_folder = 'MachineLearning/models'
    model = None
    today = int(time.time())

    def __init__(self, labels: list) -> None:
        self.labels = labels

    def __preprocess(
            self,
            hiscores: data_class, 
            players: dict=None, 
            labels: dict=None
        ):
        train = not( None == players == labels)
        player = players[0] if players else []
        logging.debug(f'Preprocessing: {train=}, {player=}, {labels=}')
        if train:
            # players datafrmae
            df_players = pd.DataFrame(players)
            df_players.set_index('id', inplace=True)
            # filter dataframe
            df_players = df_players[['label_id']]

            # labels dataframe
            df_labels = pd.DataFrame(labels)
            df_labels.set_index('id', inplace=True)
            # memory cleanup
            del players, labels
        
        # hiscores dataframe
        df_hiscores = hiscores.filter_features(base=False, feature=True, ratio=True)

        # merge dataframes
        df = df_hiscores.copy()
        if train: 
            df = df.merge(df_players, left_index=True, right_index=True)
            df = df.merge(df_labels, left_on='label_id', right_index=True)
            # cleanup columns
            df.drop(columns=['label_id'], inplace=True)
            
        # Create x & y data
        x = df[df_hiscores.columns]
        y = df['label'] if train else None

        # train test split
        if train:
            # X_train, X_test, y_train, y_test
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
            return (train_x, train_y, test_x, test_y)
        return (x, y)

    def __best_file_path(self, startwith):
        files = []
        for f in os.listdir(self.save_folder):
            if f.endswith(".joblib") and f.startswith(startwith):
                # accuracy is in filename, so we have to parse it
                model_file = f.replace('.joblib','')
                model_file = model_file.split(sep='_')
                # save to dict
                d ={
                    'path': f'{self.save_folder}/{f}',
                    'model': model_file[0],
                    'date': model_file[1],
                    'accuracy': model_file[2]
                }
                # add dict to array
                files.append(d)

        # array of dict can be used for pandas dataframe
        df_files = pd.DataFrame(files)
        df_files.sort_values(by=['date'], ascending=False, inplace=True)
        model_path = df_files['path'].iloc[0]
        return model_path

    def load(self, name) -> None:
        try:
            path = self.__best_file_path(name)
            logging.debug(f'{name=}, {path=}')
            object = load(path)
        except Exception as exception:
            logging.warning(f'Error when loading {name}: {exception=}')
            return
        return object

    def __save(self, object, name, score):
        #TODO save model & other objects we might need
        dump(value=object, filename=f'{self.save_folder}/{name}_{self.today}_{score}.joblib')
        return

    def create_model(self):
        logging.debug('Creating Model')
        self.model = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
        return self.model
    
    def train(self, players: dict, labels: dict, hiscores: data_class) -> None:
        # preprocess data
        (train_x, train_y, test_x, test_y) = self.__preprocess(
            hiscores=hiscores, players=players, labels=labels
        )

        # memory cleanup
        del players, labels, hiscores

        # create model if there is none
        if self.model == None:
            self.create_model()

        # fit model
        logging.debug('Training Model')
        self.model.fit(train_x, train_y)

        # evaluate model
        logging.debug('Scoring Model')
        score = self.model.score(test_x, test_y)
        score = round(score, 2)

        # logging
        logging.debug(f'MachineLearning: {score=}')
        # save model
        self.__save(self.model, 'model', score)
        return # evaluation

    def predict(self, hiscores: data_class) -> pd.DataFrame:
        # preprocess data
        (x, y) = self.__preprocess(
            hiscores=hiscores, players=None, labels=None
        )

        # load model if not exists
        if self.model == None:
            self.model = self.load('model')

        # make prediction
        proba =     self.model.predict_proba(x)
        proba_max = proba.max(axis=1)
        pred =      self.model.predict(x)

        df_proba =          pd.DataFrame(proba,         index=x.index, columns=self.model.classes_).round(4)
        df_proba_max =      pd.DataFrame(proba_max,     index=x.index, columns=['Predicted_confidence']).round(4)
        df_predictions =    pd.DataFrame(pred,          index=x.index, columns=['Prediction'])

        df_proba_max = df_proba_max*100
        df_proba = df_proba*100

        df = pd.DataFrame(index=x.index)
        df['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        df = df.merge(df_predictions, left_index=True, right_index=True, suffixes=('', '_prediction'), how='inner')
        df = df.merge(df_proba_max,   left_index=True, right_index=True, how='inner')
        df = df.merge(df_proba,       left_index=True, right_index=True, suffixes=('', '_probability'), how='inner')
        df = df.merge(hiscores.users, left_index=True, right_on='Player_id', how='inner')
        
        mask = (df['Player_id'].isin(hiscores.df_low.index))
        df.loc[mask, ['Prediction']] = 'Stats too low'

        df.rename(columns={'Player_id':'id'}, inplace=True)
        return df
