import asyncio
import logging

import requests
from fastapi import HTTPException, BackgroundTasks

from config import app, detector_api, secret_token, token
from MachineLearning.data import data_class
from MachineLearning.model import model

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
    'Clue_Scroll_bot',
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

async def loop_request(base_url, json={}, type='get'):
    '''
        this function gets all the data of a paginated route
    '''
    i, data = 1, []

    while True:
        # build url
        url = f'{base_url}&row_count=100000&page={i}'

        # logging
        logging.debug(f'Request: {url=}')

        # make reqest
        if type == 'get':
            res = requests.get(url)
        elif type == 'post':
            res = requests.post(url, json=json)
            logging.debug(json)
        else:
            raise 'No type specified'
        
        # escape condition
        if not res.status_code == 200:
            logging.debug(f'Break: {res.status_code=}, {url=}')
            logging.debug(res.text)
            break
        
        # parse data
        res = res.json()
        data += res

        # escape condition
        if len(res) == 0:
            logging.debug(f'Break: {len(res)=}, {url=}')
            break
        
        # logging (after break)
        logging.debug(f'Succes: {len(res)=}, {len(data)=} {i=}')

        # update iterator
        i += 1
    return data


async def stage_and_train(token: str):
    # request labels
    url = f'{detector_api}/v1/label?token={token}'

    # logging
    logging.debug(f'Request: {url=}')
    data = requests.get(url).json()

    # filter labels
    labels = [d for d in data if d['label'] in LABELS]
    
    # memory cleanup
    del url, data

    # create an input dict for url
    players = []
    for label in labels:
        url = f'{detector_api}/v1/player?token={token}&label_id={label["id"]}'
        players.extend(await loop_request(url))

    # memory cleanup
    del url

    hiscores = []
    for player in players:
        url = f'{detector_api}/v1/hiscore/Latest?token={token}&player_id={player["id"]}'
        hiscores.extend(await loop_request(url))


    # hiscores dict
    hiscores = data_class(hiscores)

    # memory cleanup
    del url

    ml.train(players, labels, hiscores)

    logging.debug("Preparing to clean training data.")
    del players, labels, hiscores
    logging.debug("Training data cleanup completed.")

    return

async def get_player_hiscores():
    logging.debug('getting data')
    url = f'{detector_api}/v1/prediction/data?token={token}&limit=50000'
    logging.debug(url)
    data = requests.get(url).json()

    # if there is no data wait and try to see if there is new data
    if len(data) == 0:
        logging.debug('no data to predict')
        await asyncio.sleep(600)
        return asyncio.create_task(get_player_hiscores())

    # clean & filter data
    data = data_class(data)
    
    # make predictions
    predictions = ml.predict(data) # dataframe
    predictions = predictions.to_dict(orient='records') # list of dict

    # post predictions
    url = f'{detector_api}/v1/prediction?token={token}'
    logging.debug(url)
    resp = requests.post(url, json=predictions)

    print(resp.text)
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
async def train(secret: str, token: str, train_tasks: BackgroundTasks):
    #TODO: verify token
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    train_tasks.add_task(stage_and_train, token)

    return {'ok': 'Training has begun.'}
