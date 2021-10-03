from logging import log
import logging
import numpy as np
import pandas as pd



skills = [
    "attack","defence","strength","hitpoints","ranged","prayer","magic","cooking","woodcutting","fletching","fishing","firemaking","crafting","smithing","mining","herblore","agility","thieving","slayer","farming","runecraft","hunter","construction"
]

class data_class:
    def __init__(self, data) -> None:
        self.df = pd.DataFrame(data)

        # defaults
        self.df_clean = None
        self.skills = skills

    def clean(self):
        logging.debug('Cleaning data')
        self.df_clean = self.df.copy()
        
        # drop unrelevant columns
        self.users = self.df_clean[['Player_id','name']]
        self.df_clean.drop(columns=['id','timestamp','ts_date','name'], inplace=True)

        # set unique index
        self.df_clean.set_index(['Player_id'], inplace=True)

        columns = self.df_clean.columns
        self.minigames = [c for c in columns if c not in skills]

        # total is not always on hiscores
        self.df_clean[self.skills] = self.df_clean[self.skills].replace(-1, 1)
        self.df_clean['total'] = self.df_clean[self.skills].sum(axis=1)
        
        self.df_clean[self.minigames] = self.df_clean[self.minigames].replace(-1, 1)
        self.df_clean['boss_total'] = self.df_clean[self.minigames].sum(axis=1)
        return self.df_clean
    
    def add_features(self):
        logging.debug('adding features')
        if self.df_clean == None:
            self.clean()
        
        # save total column to variable
        total = self.df_clean['total']
        boss_total =  self.df_clean['boss_total']

        # for each skill, calculate ratio
        for skill in self.skills:
            self.df_clean[f'{skill}/total'] = self.df_clean[skill] / total

        for boss in self.minigames:
            self.df_clean[f'{boss}/boss_total'] = self.df_clean[boss] / boss_total

        self.df_clean['median_feature'] = self.df_clean[self.skills].median(axis=1)
        self.df_clean['mean_feature'] = self.df_clean[self.skills].mean(axis=1)

        # replace infinities & nan
        self.df_clean = self.df_clean.replace([np.inf, -np.inf], 0) 
        self.df_clean.fillna(0, inplace=True)
        self.features = True
        return self.df_clean

    def filter_features(self, base:bool=True, feature:bool=True, ratio:bool=True):
        logging.debug(f'filtering features: {base=}, {feature=}, {ratio=}')
        
        # input validation
        if not(base or feature or ratio):
            raise 'pick at least one filter'

        # if the data is not cleaned, clena the data first
        if self.df_clean == None:
            self.add_features() if feature else self.clean()
        
        # filters
        base_columns = self.df.columns if base else []
        feature_columns = [c for c in self.df_clean.columns if '_feature' in c] if feature else []
        ratio_columns = [c for c in self.df_clean.columns if '/total' in c or '/boss_total'] if ratio else []

        # combine all columns
        columns = base_columns + feature_columns + ratio_columns
        return self.df_clean[columns]

