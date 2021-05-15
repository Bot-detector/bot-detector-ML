from joblib import dump, load
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV



def create_model(train_x, train_y, test_x, test_y, lbls):
    # commented for performance reasons
    # neigh = KNeighborsClassifier(n_neighbors=len(lbls), n_jobs=-1)
    # neigh = neigh.fit(train_x, train_y)

    # mlpc = MLPClassifier(max_iter=10000, random_state=7)
    # mlpc = mlpc.fit(train_x, train_y)

    # rfc = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)
    # rfc = rfc.fit(train_x, train_y)

    # etc = ExtraTreesClassifier(n_estimators=100, random_state=7, n_jobs=-1)
    # etc = etc.fit(train_x, train_y)

    # sgdc = SGDClassifier(max_iter=1000, tol=1e-3, loss='modified_huber')
    # sgdc = sgdc.fit(train_x, train_y)

    # models = [neigh, mlpc, rfc, etc, sgdc]
    # scores = [round(m.score(test_x, test_y)*100,2) for m in models]
    # weights = [s**2 for s in scores]
    # estimators = [(m.__class__.__name__, m) for m in models]

    # _ = [Config.debug(f'Model: {m.__class__.__name__} Score: {s}') for m, s in zip(models,scores)]

    # vote = VotingClassifier(
    #     weights=weights,
    #     estimators=estimators, 
    #     voting='soft',
    #     n_jobs=-1
    #     )
    # return vote
    rfc = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)
    rfc = rfc.fit(train_x, train_y)
    return rfc


    
def train_model(n_pca):
    
    df =            pf.get_highscores()
    df_players =    pf.get_players()
    df_labels =     pf.get_labels() 

    # pandas pipeline
    df_clean = (df
        .pipe(pf.start_pipeline)
        .pipe(pf.clean_dataset, ed.skills_list, ed.minigames_list)
        .pipe(pf.f_features,    ed.skills_list, ed.minigames_list)
        # .pipe(pf.filter_relevant_features, ed.skills_list)
    )
    df_preprocess = (df_clean
        .pipe(pf.start_pipeline)
        .pipe(pf.f_standardize)
        .pipe(pf.f_normalize)
    )


    today = time.strftime('%Y-%m-%d', time.gmtime())
    columns = df_preprocess.columns.tolist()
    dump(value=columns, filename=f'Predictions/models/features_{today}_100.joblib')
    

    df_pca, pca_model = pf.f_pca(df_preprocess, n_components=n_pca, pca=None)
    dump(value=pca_model, filename=f'Predictions/models/pca_{today}_{n_pca}.joblib')

    df_pca = df_preprocess # no pca
    Config.debug(f'pca shape: {df_pca.shape}')

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

    Config.debug(f'labels: {len(lbls)}, {lbls}')

    # creating x, y data, with players that a label
    mask = ~(df_pca['label_id'] == 0) & (df_pca['label'].isin(lbls))
    df_labeled = df_pca[mask].copy()
    df_labeled.drop(columns=['confirmed_ban','confirmed_player','possible_ban','label_id'], inplace=True)
    x, y = df_labeled.iloc[:,:-1], df_labeled.iloc[:,-1]

    # save labels
    lbls = np.sort(y.unique())
    dump(value=lbls, filename=f'Predictions/models/labels_{today}_100.joblib')

    # train test split but make sure to have all the labels form y
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    model_name = 'vote'
    model = create_model(train_x, train_y, test_x, test_y, lbls)
    model = model.fit(train_x, train_y)

    # works on colab not on my pc: ValueError: Invalid prediction method: _predict_proba 
    # # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV
    # does not work in current version, issue created https://github.com/scikit-learn/scikit-learn/issues/20053
    # model = CalibratedClassifierCV(base_estimator=model, cv='prefit')
    # model = model.fit(test_x, test_y) # docu says to calibrate on test?

    # print model score
    model_score = round(model.score(test_x, test_y)*100,2)
    Config.debug(f'Score: {model_score}')

    # print more detailed model score
    Config.debug(classification_report(test_y, model.predict(test_x), target_names=lbls))

    # fit & save model on entire dataset
    model = model.fit(x, y)
    dump(value=model, filename=f'Predictions/models/model-{model_name}_{today}_{model_score}.joblib')
    return None