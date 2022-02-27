import asyncio
import logging
import time

import numpy as np
import pandas as pd
import requests
from fastapi import HTTPException
from sklearn.model_selection import train_test_split

from api import config
from api.cogs import requests as req
from api.MachineLearning import classifier, data

logger = logging.getLogger(__name__)

app = config.app

binary_classifier = classifier.classifier('binaryClassifier').load()
multi_classifier = classifier.classifier('multiClassifier').load()


@app.on_event('startup')
async def initial_task():

    return


@app.get("/")
async def root():
    return {"detail": "hello world"}


@app.get("/startup")
async def manual_startup(secret: str):
    """
        start predicting
    """
    # secret token for api's to talk to eachother
    if secret != config.secret_token:
        raise HTTPException(
            status_code=404, detail=f"insufficient permissions")

    while True:
        # endpoint that we are going to use
        data_url = f'{config.detector_api}/v1/prediction/data?token={config.token}&limit=100'
        output_url = f'{config.detector_api}/v1/prediction?token={config.token}'

        hiscores = req.request([data_url])
        hiscores = pd.DataFrame(hiscores)
        if len(hiscores) == 0:
            logger.debug('No data: sleeping')
            time.sleep(60)
            continue

        names = hiscores[["id", "name"]]
        hiscores = hiscores[[c for c in hiscores.columns if c != "name"]]

        hiscores = data.hiscoreData(hiscores)
        hiscores = hiscores.features()

        # binary prediction
        binary_pred = binary_classifier.predict_proba(hiscores)
        binary_pred = pd.DataFrame(
            binary_pred,
            index=hiscores.index,
            columns=['Real_Player', 'Unknown_bot']
        )

        # multi prediction
        multi_pred = multi_classifier.predict_proba(hiscores)
        multi_pred = pd.DataFrame(
            multi_pred, index=hiscores.index, columns=np.unique(config.LABELS)
        )

        # combine binary & player_pred
        output = pd.DataFrame(names).set_index("id")
        output = output.merge(
            binary_pred, left_index=True, right_index=True, how='left'
        )
        output = output.merge(
            multi_pred, left_index=True, right_index=True, suffixes=['', '_multi'], how="left"
        )

        # cleanup predictions
        mask = (output["Real_Player"].isna())
        output.loc[
            output["Real_Player"].isna(), "Unknown_bot"
        ] = output[mask]["Real_Player_multi"]

        output.drop(columns=["Real_Player_multi"], inplace=True)
        output.fillna(0, inplace=True)

        # add Predictions & Predicted_confidence
        columns = [c for c in output.columns if c != "name"]
        output['Predicted_confidence'] = output[columns].max(axis=1)
        output["Prediction"] = output[columns].idxmax(axis=1)

        # post output
        output = output.to_dict(orient='records')
        print(output[:2])
        requests.post(output_url, json=output)
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
