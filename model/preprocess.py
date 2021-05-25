import datetime as dt
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, RobustScaler


def logging(f):
    def wrapper(df, *args, **kwargs):
        start = dt.datetime.now()
        result = f(df, *args, **kwargs)
        end = dt.datetime.now()
        try:
            print(f'{f.__name__} took: {end - start} shape= {result.shape}')
        except:
            print(f'{f.__name__} took: {end - start}')
        return result
    return wrapper

@logging
def start_pipeline(df):
    return df.copy()

@logging
def clean_dataset(df, skills_list, minigames_list):
    # sort by timestamp, drop duplicates keep last
    df = df.sort_values('timestamp').drop_duplicates('Player_id',keep='last')

    # drop unrelevant columns
    df.drop(columns=['id','timestamp','ts_date','Player_id'], inplace=True)

    # set unique index
    df.set_index(['name'],inplace=True)

    # total is sum of all skills
    df['total'] = df[skills_list].sum(axis=1)
    

    # replace -1 values
    df[skills_list] = df[skills_list].replace(-1, 1)
    df[minigames_list] = df[minigames_list].replace(-1, 0)

    df['boss_total'] = df[minigames_list].sum(axis=1)

    return df

def wintertodt_feature(df):
    wintertodt_fm = df['wintertodt']*30_000
    lvl50skill = 101_333

    df['wintertodt_feature']        = wintertodt_fm/(df['firemaking'] - lvl50skill)

    df['wintertodt_lag_feature']    = np.where(df['wintertodt'] > 670, 1, 0)
    return df

def botname(df):
    mask = (df.index.astype(str).str[0:2].str.isdigit())
    df['botname_feature'] = 0
    df.loc[mask,'botname_feature'] = 1

    return df

@logging
def f_features(df, skills_list, minigames_list):
    # save total column to variable
    total = df['total']
    boss_total =  df['boss_total']

    # for each skill, calculate ratio
    for skill in skills_list:
        df[f'{skill}/total'] = df[skill] / total

    for boss in minigames_list:
        df[f'{boss}/boss_total'] = df[boss] / boss_total

    # add custom features
    df = wintertodt_feature(df)
    df = botname(df)

    df['median_feature'] = df[skills_list].median(axis=1)
    df['mean_feature'] = df[skills_list].mean(axis=1)
    df['std_feature'] = df[skills_list].std(axis=1)

    # cleanup infinities
    df = df.replace([np.inf, -np.inf], 0) 
    df.fillna(0, inplace=True)

    return df

@logging
def f_standardize(df, scaler=None):
    if scaler is None:
        print('new scaler')
        scaler = RobustScaler()
        scaler = scaler.fit(df)

        today = int(time.time())
        dump(value=scaler, filename=f'model/models/scaler_{today}_100.joblib')
    
    X_scaled = scaler.transform(df) 
    return pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

@logging
def f_normalize(df, transformer=None):
    if transformer is None:
        print('new normalizer')
        transformer = Normalizer()
        transformer = transformer.fit(df)

        today = int(time.time())
        dump(value=transformer, filename=f'model/models/normalizer_{today}_100.joblib')

    X_normalized = transformer.transform(df)
    return pd.DataFrame(X_normalized, columns=df.columns, index=df.index)

def f_pca(df, n_components=2, pca=None):
    if pca is None:
        pca = PCA(n_components='mle', random_state=7) 
        pca = pca.fit(df)

        today = int(time.time())
        n_components = pca.n_components_
        dump(value=pca, filename=f'model/models/pca_{today}_{n_components}.joblib')
        
    # Apply dimensionality reduction to X.
    X_principal = pca.transform(df)
    # rename columns and put in dataframe
    columns = [f'P{c}' for c in range(n_components)]
    df = pd.DataFrame(X_principal, columns=columns, index=df.index) 
    df.dropna(inplace=True)
    return df, pca
