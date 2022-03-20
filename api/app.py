import asyncio
import logging
import time
from typing import List

import pandas as pd
import requests
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from api import config
from api.cogs import predict
from api.cogs import requests as req
from api.MachineLearning import classifier, data

logger = logging.getLogger(__name__)

app = config.app

binary_classifier = classifier.classifier('binaryClassifier').load()
multi_classifier = classifier.classifier('multiClassifier').load()


class name(BaseModel):
    id: int
    name: str


@app.on_event('startup')
async def initial_task():
    global binary_classifier, multi_classifier
    if binary_classifier is None or multi_classifier is None:
        binary_classifier = classifier.classifier('binaryClassifier')
        multi_classifier = classifier.classifier('multiClassifier')
        await train(config.secret_token)
    await manual_startup(config.secret_token)
    return


@app.get("/")
async def root():
    return {"detail": "hello worldz"}


@app.get("/startup")
async def manual_startup(secret: str):
    """
        start predicting
    """
    # secret token for api's to talk to eachother
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions"
        )


    while True:
        # endpoint that we are going to use
        data_url = f'{config.detector_api}/v1/prediction/data?token={config.token}&limit={config.BATCH_AMOUNT}'
        output_url = f'{config.detector_api}/v1/prediction?token={config.token}'

        hiscores = req.request([data_url])
        hiscores = pd.DataFrame(hiscores)

        if len(hiscores) == 0:
            logger.debug('No data: sleeping')
            time.sleep(600)
            continue

        names = hiscores[["Player_id", "name"]]
        names = names.rename(columns={"Player_id": "id"})
        hiscores = hiscores[[c for c in hiscores.columns if c != "name"]]

        output = predict.predict(
            hiscores, 
            names, 
            binary_classifier, 
            multi_classifier
        )
        
        logger.debug("Sending response")
        resp = requests.post(output_url, json=output)

        if resp.status_code != 200:
            print(resp.text[0])
            time.sleep(600)
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
async def predict_player(secret: str, hiscores, name: name) -> List[dict]:
    """
        predict one player
    """
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")
    name = pd.DataFrame(name.dict())
    output = predict.predict(
        hiscores, name, binary_classifier, multi_classifier)
    return output


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
    labels = req.request([label_url])

    # request players
    player_urls = [
        f'{player_url}&label_id={label["id"]}' for label in labels
        if label["label"] in config.LABELS
    ]
    players = req.request(player_urls)

    # request hiscore data
    hiscore_urls = [
        f'{hiscore_url}&label_id={label["id"]}' for label in labels
        if label["label"] in config.LABELS
    ]
    hiscores = req.request(hiscore_urls)

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
