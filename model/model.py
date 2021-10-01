import os
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, RobustScaler


class data:
    # not a big fan of how i set this up
    minigames = {
        "league": "",
        "bounty_hunter_hunter": "",
        "bounty_hunter_rogue": "",
        "cs_all": "",
        "cs_beginner": "",
        "cs_easy": "",
        "cs_medium": "",
        "cs_hard": "",
        "cs_elite": "",
        "cs_master": "",
        "lms_rank": "",
        "soul_wars_zeal": "",
        "abyssal_sire": "",
        "alchemical_hydra": "",
        "barrows_chests": "",
        "bryophyta": "",
        "callisto": "",
        "cerberus": "",
        "chambers_of_xeric": "",
        "chambers_of_xeric_challenge_mode": "",
        "chaos_elemental": "",
        "chaos_fanatic": "",
        "commander_zilyana": "",
        "corporeal_beast": "",
        "crazy_archaeologist": "",
        "dagannoth_prime": "",
        "dagannoth_rex": "",
        "dagannoth_supreme": "",
        "deranged_archaeologist": "",
        "general_graardor": "",
        "giant_mole": "",
        "grotesque_guardians": "",
        "hespori": "",
        "kalphite_queen": "",
        "king_black_dragon": "",
        "kraken": "",
        "kreearra": "",
        "kril_tsutsaroth": "",
        "mimic": "",
        "nightmare": "",
        "obor": "",
        "sarachnis": "",
        "scorpia": "",
        "skotizo": "",
        "the_gauntlet": "",
        "the_corrupted_gauntlet": "",
        "theatre_of_blood": "",
        "thermonuclear_smoke_devil": "",
        "tzkal_zuk": "",
        "tztok_jad": "",
        "venenatis": "",
        "vetion": "",
        "vorkath": "",
        "wintertodt": "",
        "zalcano": "",
        "zulrah": ""
    }

    skills = {
        "total": "",
        "Attack": "",
        "Defence": "",
        "Strength": "",
        "Hitpoints": "",
        "Ranged": "",
        "Prayer": "",
        "Magic": "",
        "Cooking": "",
        "Woodcutting": "",
        "Fletching": "",
        "Fishing": "",
        "Firemaking": "",
        "Crafting": "",
        "Smithing": "",
        "Mining": "",
        "Herblore": "",
        "Agility": "",
        "Thieving": "",
        "Slayer": "",
        "Farming": "",
        "Runecraft": "",
        "Hunter": "",
        "Construction": "",
    }

    skills_list = [item.lower() for item in skills.keys()]
    skills_list.remove('total')

    minigames_list = [m.lower() for m in minigames.keys()]

    # initialised variables
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def __clean_dataset(self):
        # sort by timestamp, drop duplicates keep last
        self.df_clean = self.df.copy().sort_values('timestamp').drop_duplicates('Player_id',keep='last')

        # drop unrelevant columns
        self.df_clean.drop(columns=['id','timestamp','ts_date','Player_id'], inplace=True)

        # set unique index
        self.df_clean.set_index(['name'],inplace=True)

        # total is not always on hiscores
        self.df_clean[self.skills_list] = self.df_clean[self.skills_list].replace(-1, 1)
        self.df_clean['total'] = self.df_clean[self.skills_list].sum(axis=1)
        
        self.df_clean[self.minigames_list] = self.df_clean[self.minigames_list].replace(-1, 1)
        self.df_clean['boss_total'] = self.df_clean[self.minigames_list].sum(axis=1)
        return self.df_clean

    def __zalcano_feature(self):
        lvl70skill = 737_627
        # song of the elves requirements
        req = ['agility','construction','farming','herblore','hunter','smithing','woodcutting']
        self.df_clean['zalcano_flag_feature'] = np.where(self.df_clean[req].min(axis=1) > lvl70skill, 1, 0) 
        return

    def __botname(self):
        L = '[A-Za-z]'
        C = '[_-]'
        S = '[ ]'
        N = '[0-9]'
        name_transform = [
                        (L, 'L'), 
                        (C, 'C'), 
                        (S, 'S'), 
                        (N, 'N')]
        self.df_clean['botname'] = self.df_clean.index
        for token_regex, token in name_transform:
            self.df_clean['botname'] = self.df_clean['botname'].str.replace(token_regex, token)
        
        self.df_clean['botname'] = self.df_clean['botname'].astype("category").cat.add_categories([0])
        self.df_clean['botname_feature'] = self.df_clean['botname'].cat.codes
        return

    def __add_features(self):
        # save total column to variable
        total = self.df_clean['total']
        boss_total =  self.df_clean['boss_total']

        # for each skill, calculate ratio
        for skill in self.skills_list:
            self.df_clean[f'{skill}/total'] = self.df_clean[skill] / total

        for boss in self.minigames_list:
            self.df_clean[f'{boss}/boss_total'] = self.df_clean[boss] / boss_total

        self.__zalcano_feature()
        self.__botname()

        self.df_clean['median_feature'] = self.df_clean[self.skills_list].median(axis=1)
        self.df_clean['mean_feature'] = self.df_clean[self.skills_list].mean(axis=1)
        # df['std_feature'] = df[skills_list].std(axis=1)

        # replace infinities & nan
        self.df_clean = self.df_clean.replace([np.inf, -np.inf], 0) 
        self.df_clean.fillna(0, inplace=True)
        return self.df_clean

    def clean(self):
        self.__clean_dataset()
        self.__add_features()
        return self.df_clean

class model:
    save_folder = 'model/models'
    scaler = None
    normalizer = None
    pca = None
    features = None
    model = None
    today = int(time.time())
    pca_components = 'mle'

    labels = [
        'Real_Player', 'Smithing_bot', 'Mining_bot', 
        'Magic_bot', 'PVM_Ranged_bot',  
        'Fletching_bot', 'PVM_Melee_bot', 'Herblore_bot',
        'Thieving_bot','Crafting_bot', 'PVM_Ranged_Magic_bot',
        'Hunter_bot','Runecrafting_bot','Fishing_bot','Agility_bot',
        'Cooking_bot', 'mort_myre_fungus_bot', 
        'Woodcutting_bot', 'Fishing_Cooking_bot',
        'Agility_Thieving_bot', 'Construction_Magic_bot','Construction_Prayer_bot',
        'Zalcano_bot'
    ]

    def __init__(self, df) -> None:
        self.df = df
        
    def __preprocess(self, scale=False, normalize=False, pca=False):
        # filter features
        if self.features is None:
            features = self.df.columns
            self.features = [f for f in features if '/total' in f or '/boss_total' in f or '_feature' in f]

            dump(value=self.features, filename=f'{self.save_folder}/features_{self.today}_100.joblib')
        # create scaler
        if self.scaler is None:
            self.scaler = RobustScaler()
            self.scaler = self.scaler.fit(self.df)

            dump(value=self.scaler, filename=f'{self.save_folder}/scaler_{self.today}_100.joblib')
        # create normalizer
        if self.normalizer is None:
            self.normalizer = Normalizer()
            self.normalizer = self.normalizer.fit(self.df)
            
            dump(value=self.normalizer, filename=f'{self.save_folder}/normalizer_{self.today}_100.joblib')
        # create principal component analysis
        if self.pca is None:
            self.pca_components = int(self.pca_components) if self.pca_components != 'mle' else self.pca_components
            self.pca = PCA(n_components=self.pca_components, random_state=7) 
            self.pca = self.pca.fit(self.df)

            n_components = self.pca.n_components_
            dump(value=self.pca, filename=f'{self.save_folder}/pca_{self.today}_{n_components}.joblib')

        self.df = self.df[self.features].copy()
        # transform the data
        if scale:
            self.df = self.scaler.transform(self.df)
        
        if normalize:
            self.df = self.normalizer.transform(self.df)

        if pca:
            self.df = self.pca.transform(self.df)
            n_components = self.pca.n_components_
            # rename columns and put in dataframe
            columns = [f'P{c}' for c in range(n_components)]
            self.df = pd.DataFrame(self.df, columns=columns, index=self.df.index) 

        # creating x, y data, with players that a label
        mask = ~(self.df['label_id'] == 0) & (self.df['label'].isin(self.labels))
        df_labeled = self.df[mask].copy()
        df_labeled.drop(columns=['confirmed_ban','confirmed_player','possible_ban','label_id'], inplace=True)
        self.x, self.y = df_labeled.iloc[:,:-1], df_labeled.iloc[:,-1]

        # train test split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.3, random_state=42, stratify=self.y)

        # save labels
        lbls = np.sort(self.y.unique())
        dump(value=lbls, filename=f'{self.save_folder}/labels_{self.today}_100.joblib')
        return self.train_x, self.test_x, self.train_y, self.test_y

    def train(self):
        model_name = 'rfc'
        self.model = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)
        self.model.fit(self.x, self.y)

        model_score = round(self.model.score(self.test_x, self.test_y)*100,2)
        dump(value=self.model, filename=f'{self.save_folder}/model-{model_name}_{self.today}_{model_score}.joblib')
        return self.model

    def __best_file_path(self, startwith):
        files = []
        for f in os.listdir(self.save_folder):
            if f.endswith(".joblib") and f.startswith(startwith):
                # accuracy is in filename, so we have to parse it
                model_file = f.replace('.joblib','')
                model_file = model_file.split(sep='_')
                # save to dict
                d ={
                    'path': str(f),
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
        accuracy = df_files['accuracy'].iloc[0]
        return model_path, accuracy

    def __load(self):
        # try to load latest saved models
        try:
            scaler, _ = self.__best_file_path(startwith='scaler')
            self.scaler = load(scaler)

            normalizer, _ = self.__best_file_path(startwith='normalizer')
            self.normalizer = load(normalizer)

            features, _ = self.__best_file_path(startwith='features')
            features = load(features)

            pca, self.pca_components  = self.__best_file_path(startwith='pca')
            self.pca = load(pca)

            labels, _ = self.__best_file_path(startwith='labels')
            self.labels = load(labels)

            model, _ = self.__best_file_path(startwith='model')
            self.model = load(model)
            return 1
        except Exception as e:
            #TODO: error =>
            return None

    def predict(self, df):
        # check if model object is loaded
        if self.model is None:
            succes = self.__load()

        # if load was not succesfull train model
        if succes is None:
            self.train()

        # preprocess data
        self.__preprocess(df)

        # predict data
        pred = self.model.predict(self.df)
        return pred
