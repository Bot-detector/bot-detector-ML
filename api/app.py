import logging

import pandas as pd
from fastapi import HTTPException
from sklearn.model_selection import train_test_split

from api import config
from api.cogs import requests
from api.MachineLearning import classifier, data

logger = logging.getLogger(__name__)

app = config.app

binary_classifier = classifier.classifier('binaryClassifier', n_estimators=300, n_jobs=-1)
multi_classifier = classifier.classifier('multiClassifier', n_estimators=300, n_jobs=-1)


# @app.on_event('startup')
# async def initial_task():
#     asyncio.create_task(get_player_hiscores())
#     return


@app.get("/")
async def root():
    return {"detail": "hello world"}


@app.get("/startup")
async def manual_startup(secret: str):
    """
        start predicting
    """
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")

    return {'detail': 'ok'}


@app.get("/load")
async def load(secret: str):
    global binary_classifier, multi_classifier
    """
        load the latest model
    """
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")
    binary_classifier = binary_classifier.load()
    multi_classifier = multi_classifier.load()
    return {'detail': 'ok'}


@app.get("/predict")
async def predict(secret: str):
    """
        predict one player
    """
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")

    return


@app.get("/train")
async def train(secret: str):
    """
        train a new model
    """
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")

    # api endpoints
    label_url = f'{config.detector_api}/v1/label?token={config.token}'
    player_url = f'{config.detector_api}/v1/player/bulk?token={config.token}'
    hiscore_url = f'{config.detector_api}/v1/hiscore/Latest/bulk?token={config.token}'

    # request labels
    labels = requests.request([label_url])
    logger.debug(labels[:2])

    # request players
    player_urls = [
        f'{player_url}&label_id={label["id"]}' for label in labels
        if label in config.LABELS
    ]
    players = requests.request(player_urls)

    # request hiscore data
    hiscore_urls = [
        f'{hiscore_url}&label_id={label["id"]}' for label in labels
        if label in config.LABELS
    ]
    hiscores = requests.request(hiscore_urls)

    # parse hiscoreData
    hiscoredata = data.hiscoreData(hiscores)
    del hiscores

    # get the desired features
    features = hiscoredata.features()
    del hiscoredata

    ###############################################################
    # get players with binary target
    player_data = data.playerData(players, labels).get(binary=True)

    # merge features with target
    features_labeled = features.merge(
        player_data, left_index=True, right_index=True)

    # create train test data
    x, y = features_labeled.iloc[:, :-1], features_labeled.iloc[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # train & score the model
    binary_classifier.fit(train_x, train_y)
    binary_classifier.score(test_y, test_x)

    # save the model
    binary_classifier.save()
    ###############################################################

    # get players with multi target
    player_data = data.playerData(players, labels).get(binary=False)

    # merge features with target
    features_labeled = features.merge(
        player_data, left_index=True, right_index=True
    )

    # we need at least 100 users
    to_little_data_labels = pd.DataFrame(
        features_labeled.iloc[:, -1].value_counts()
    ).query('target < 100').index
    mask = ~(features_labeled['target'].isin(to_little_data_labels))
    features_labeled = features_labeled[mask]

    # create train test data
    x, y = features_labeled.iloc[:, :-1], features_labeled.iloc[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # train & score the model
    multi_classifier.fit(train_x, train_y)
    multi_classifier.score(test_y, test_x)

    # save the model
    multi_classifier.save()
    ###############################################################

    return {'detail': 'ok'}
