# set path to 1 folder up
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import model.functions as functions
import model.preprocess as pp
import model.extra_data as ed

import time

def create_model(train_x, train_y, test_x, test_y, lbls):
    rfc = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)
    rfc = rfc.fit(train_x, train_y)
    return rfc


def train_model():
    df =            functions.get_highscores(config.token)
    df_players =    functions.get_players(config.token)
    df_labels =     functions.get_labels(config.token)

    # pandas pipeline
    df_clean = (df
        .pipe(pp.start_pipeline)
        .pipe(pp.clean_dataset, ed.skills_list, ed.minigames_list)
        .pipe(pp.f_features,    ed.skills_list, ed.minigames_list)
        # .pipe(pp.filter_relevant_features, ed.skills_list)
    )
    df_preprocess = (df_clean
        .pipe(pp.start_pipeline)
        .pipe(pp.f_standardize)
        .pipe(pp.f_normalize)
    )

    today = int(time.time()) 
    columns = df_preprocess.columns.tolist()
    dump(value=columns, filename=f'Predictions/models/features_{today}_100.joblib')
    
    # principal component analysis
    # df_pca, pca_model = pf.f_pca(df_preprocess, n_components=n_pca, pca=None)
    # dump(value=pca_model, filename=f'Predictions/models/pca_{today}_{n_pca}.joblib')
    df_pca = df_preprocess 

    df_pca = df_pca.merge(df_players,   left_index=True,    right_index=True, how='inner')
    df_pca = df_pca.merge(df_labels,    left_on='label_id', right_index=True, how='left')

    lbls = [
        'Real_Player', 'Smithing_bot', 'Mining_bot', 
        'Magic_bot', 'PVM_Ranged_bot', 'Wintertodt_bot', 
        'Fletching_bot', 'PVM_Melee_bot', 'Herblore_bot',
        'Thieving_bot','Crafting_bot', 'PVM_Ranged_Magic_bot',
        'Hunter_bot','Runecrafting_bot','Fishing_bot','Agility_bot',
        'Cooking_bot', 'FarmBird_bot', 'mort_myre_fungus_bot',
        'Woodcutting_bot'
    ]

    print(f'labels: {len(lbls)}, {lbls}')
    
    # creating x, y data, with players that a label
    mask = ~(df_pca['label_id'] == 0) & (df_pca['label'].isin(lbls))
    df_labeled = df_pca[mask].copy()
    df_labeled.drop(columns=['confirmed_ban','confirmed_player','possible_ban','label_id'], inplace=True)
    x, y = df_labeled.iloc[:,:-1], df_labeled.iloc[:,-1]
    return None

if __name__ == "__main__":
    train_model()
