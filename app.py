import asyncio
import logging

import requests
from fastapi import HTTPException

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
    'Herblore_bot'
]

ml = model(LABELS)

async def loop_request(base_url, json):
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
        res = requests.post(url, json=json)

        # escape condition
        if not res.status_code == 200:
            logging.debug(f'Break: {res.status_code=}, {url=}')
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

@app.on_event('startup')
async def initial_task():
    asyncio.create_task(get_player_hiscores())
    return

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/startup")
async def manual_startup():
    asyncio.create_task(get_player_hiscores())
    return {'ok': 'Predictions have been started.'}

@app.get("/load")
async def load(token:str):
    #TODO: verify token
    if token != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    if ml.model is None:
        ml.model = ml.load('model')
    return {'ok': 'ok'}

@app.get("/predict")
async def predict(token:str):
    #TODO: verify token
    if token != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

    return 

@app.get("/train")
async def train(secret: str, token: str):
    #TODO: verify token
    if secret != secret_token:
        raise HTTPException(status_code=404, detail=f"insufficient permissions")

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
    label_input = {}
    label_input['label_id'] = [l['id'] for l in labels]
    
    # request all player with label
    url = f'{detector_api}/v1/player/bulk?token={token}'
    data = await loop_request(url, label_input)

    # players dict
    players = data

    # memory cleanup
    del url, data

    # get all player id's
    player_input = {}
    player_input['player_id'] = [int(p['id']) for p in players]

    # request hiscore data latest with player_id
    url = f'{detector_api}/v1/hiscore/Latest/bulk?token={token}'
    data = await loop_request(url, player_input)

    # hiscores dict
    hiscores = data_class(data)

    # memory cleanup
    del url, data

    ml.train(players, labels, hiscores)
    # memory cleanup
    del players, labels, hiscores
    return {'ok': 'ok'}
