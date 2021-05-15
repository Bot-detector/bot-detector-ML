import datetime as dt
import numpy as np

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
    df[skills_list] = df[skills_list].div(total, fill_value=0)
    df[minigames_list] = df[skills_list].div(boss_total, fill_value=0)

    df = wintertodt_feature(df)
    # df = zalcano_feature(df)
    # df = botname(df)
    # df['rangebot_feature'] = (df['ranged'] + df['hitpoints'])/total

    df['median_feature'] = df[skills_list].median(axis=1)
    df['mean_feature'] = df[skills_list].mean(axis=1)
    df['std_feature'] = df[skills_list].std(axis=1)

    # replace infinities & nan
    df = df.replace([np.inf, -np.inf], 0) 
    df.fillna(0, inplace=True)

    return df
