import asyncio
import logging
from datetime import date
from typing import List

import pandas as pd
from api import config
from api.cogs import predict
from api.cogs import requests as req
from api.MachineLearning import classifier, data
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

app = config.app

binary_classifier = classifier.classifier("binaryClassifier").load()
multi_classifier = classifier.classifier("multiClassifier").load()


class name(BaseModel):
    id: int
    name: str


logger = logging.getLogger(__name__)


@app.on_event("startup")
async def initial_task():
    """
    This function is called when the api starts up.
    It will load the latest model and start the prediction process.
    """
    global binary_classifier, multi_classifier
    if binary_classifier is None or multi_classifier is None:
        binary_classifier = classifier.classifier("binaryClassifier")
        multi_classifier = classifier.classifier("multiClassifier")
        await train(config.secret_token)
    await manual_startup(config.secret_token)
    return


@app.get("/")
async def root():
    """
    This endpoint is used to check if the api is running.
    """
    return {"detail": "hello world"}


@app.get("/startup")
async def manual_startup(secret: str):
    logger.debug("manual startup")
    """
        This endpoint is used to manually start the prediction process.
        It is used by the detector api to start the prediction process.
    """
    # secret token for api's to talk to eachother
    if secret != config.secret_token:
        raise HTTPException(status_code=404, detail="insufficient permissions")

    id = 0
    today = date.today()
    while True:
        if today != date.today():
            logger.info("new day")
            id, today = 0, date.today()

        hiscores = await req.get_prediction_data(
            player_id=id, limit=config.BATCH_AMOUNT
        )
        _highscores = hiscores[-1]
        logger.info(_highscores)
        id = _highscores.get("player_id")
        hiscores = pd.DataFrame(hiscores)

        if len(hiscores) == 0:
            logger.debug("No data: sleeping")
            await asyncio.sleep(60)
            continue

        names = hiscores[["player_id"]]
        names = names.rename(columns={"player_id": "id"})
        hiscores = hiscores[[c for c in hiscores.columns if c != "name"]]

        output = predict.predict(hiscores, names, binary_classifier, multi_classifier)

        logger.debug("Sending response")
        await req.post_prediction(output)

        if len(hiscores) < config.BATCH_AMOUNT:
            sleep = 60
            logger.info(f"{len(hiscores)=} < {config.BATCH_AMOUNT=}, sleeping: {sleep}")
            await asyncio.sleep(sleep)
    return {"detail": "ok"}


@app.get("/load")
async def load(secret: str):
    logger.debug("loading model")
    global binary_classifier, multi_classifier
    """
        load the latest model.
        This endpoint is used by the detector api to load the latest model.
    """
    if secret != config.secret_token:
        raise HTTPException(status_code=404, detail="insufficient permissions")

    binary_classifier = binary_classifier.load()
    multi_classifier = multi_classifier.load()
    return {"detail": "ok"}


@app.get("/predict")
async def predict_player(secret: str, hiscores, name: name) -> List[dict]:
    """
    predict one player.
    This endpoint is used by the detector api to predict one player.
    """
    logger.debug(f"predicting player {name}")
    if secret != config.secret_token:
        raise HTTPException(status_code=404, detail="insufficient permissions")
    name = pd.DataFrame(name.dict())
    output = predict.predict(hiscores, name, binary_classifier, multi_classifier)
    return output


@app.get("/train")
async def train(secret: str):
    """
    train a new model.
    This endpoint is used by the detector api to train a new model.
    """
    logger.debug("training model")
    if secret != config.secret_token:
        raise HTTPException(status_code=404, detail="insufficient permissions")

    labels = await req.get_labels()
    players = []
    hiscores = []

    for label in labels:
        if label["label"] not in config.LABELS:
            continue

        player_data = await req.get_player_data(label_id=label["id"])
        players.extend(player_data)

        hiscore_data = await req.get_hiscore_data(label_id=label["id"])
        hiscores.extend(hiscore_data)

    # parse hiscoreData
    hiscoredata = data.hiscoreData(hiscores)
    del hiscores

    # get the desired features
    features = hiscoredata.features()
    del hiscoredata

    # get players with binary target
    player_data = data.playerData(players, labels).get(binary=True)

    # merge features with target
    features_labeled = features.merge(player_data, left_index=True, right_index=True)

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

    # get players with multi target
    player_data = data.playerData(players, labels).get(binary=False)

    # merge features with target
    features_labeled = features.merge(player_data, left_index=True, right_index=True)
    # Get the last column from the DataFrame
    last_column = features_labeled.iloc[:, -1]

    # Convert non-numeric values to NaN
    last_column_numeric = pd.to_numeric(last_column, errors="coerce")

    # Replace NaN values with 0
    last_column_numeric_filled = last_column_numeric.fillna(0)

    # Convert the series to an integer data type
    last_column_numeric_int = last_column_numeric_filled.astype(int)

    # Count the number of occurrences of each value
    value_counts = last_column_numeric_int.value_counts()

    # Convert the Series to a DataFrame
    value_counts_df = pd.DataFrame(value_counts)

    # Filter the DataFrame to get the labels with less than 100 occurrences
    to_little_data_labels = value_counts_df.query("target < 100").index

    mask = ~(features_labeled["target"].isin(to_little_data_labels))
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
    return {"detail": "ok"}
