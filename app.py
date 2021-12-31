import asyncio
import logging

import aiohttp
from fastapi import BackgroundTasks, HTTPException

from cogs import requests
from config import app, detector_api, secret_token, token
from MachineLearning.data import data_class
from MachineLearning.model import model

logger = logging.getLogger(__name__)

LABELS = [
    'Real_Player',
    'PVM_Melee_bot',
    'Smithing_bot',
    'Magic_bot',
    'Fishing_bot',
    'Mining_bot',
    'Crafting_bot',
    'PVM_Ranged_Magic_bot',
    'PVM_Ranged_bot',
    'Hunter_bot',
    'Fletching_bot',
    # 'Clue_Scroll_bot',
    'LMS_bot',
    'Agility_bot',
    'Wintertodt_bot',
    'Runecrafting_bot',
    'Zalcano_bot',
    'Woodcutting_bot',
    'Thieving_bot',
    'Soul_Wars_bot',
    'Cooking_bot',
    'Vorkath_bot',
    'Barrows_bot',
    'Herblore_bot',
    'Zulrah_bot'
]

ml = model(LABELS)


async def stage_and_train(token: str):
    # request labels
    url = f'{detector_api}/v1/label?token={token}'

    # logger
    async with aiohttp.ClientSession() as session:
        data = await requests.get_request(session, url)

    # filter labels
    labels = [d for d in data if d['label'] in LABELS]
    
    # memory cleanup
    del url, data

    # get players
    base = f'{detector_api}/v1/player/bulk?token={token}'
    urls = [f'{base}&label_id={label["id"]}' for label in labels]
    players = await requests.batch_request(urls)

    # memory cleanup
    del base, urls

    # get hiscores
    base = f'{detector_api}/v1/hiscore/Latest/bulk?token={token}'
    urls = [f'{base}&label_id={label["id"]}' for label in labels]
    hiscores = await requests.batch_request(urls)

    # memory cleanup
    del base, urls

    # hiscores dict
    hiscores = data_class(hiscores)

    ml.train(players, labels, hiscores)

    # memory cleanup
    del players, labels, hiscores
    logger.debug('ML trained')

    return

async def get_player_hiscores():
    logger.debug('getting data')
    url = f'{detector_api}/v1/prediction/data?token={token}&limit=50000'
    logger.debug(url)
    
    async with aiohttp.ClientSession() as session:
        data = await requests.get_request(session, url)
        
    # if there is no data wait and try to see if there is new data
    if len(data) == 0:
        logger.debug('no data to predict')
        await asyncio.sleep(600)
        return asyncio.create_task(get_player_hiscores())

    # clean & filter data
    data = data_class(data)
    
    # make predictions
    predictions = ml.predict(data) # dataframe
    predictions = predictions.to_dict(orient='records') # list of dict

    # post predictions
    url = f'{detector_api}/v1/prediction?token={token}'
    logger.debug(url)
    async with aiohttp.ClientSession() as session:
        resp = await requests.post_request(session, url, predictions)

    return asyncio.create_task(get_player_hiscores())

# @app.on_event('startup')
# async def initial_task():
#     asyncio.create_task(get_player_hiscores())
#     return

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/startup")
async def manual_startup(secret:str):
    #TODO: verify token
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    asyncio.create_task(get_player_hiscores())
    return {'ok': 'Predictions have been started.'}

@app.get("/load")
async def load(secret:str):
    #TODO: verify token
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    if ml.model is None:
        ml.model = ml.load('model')
    return {'ok': 'ok'}

@app.get("/predict")
async def predict(secret:str):
    #TODO: verify token
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    return 

@app.get("/train")
async def train(secret: str, token: str, background: BackgroundTasks):
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    background.add_task(stage_and_train, token)

    return {'ok': 'Training has begun.'}
