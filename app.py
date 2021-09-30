import asyncio
import logging

import aiohttp
import requests

from config import app, detector_api, token
from MachineLearning.data import data_class
from MachineLearning.model import model

LABELS = [
    'Real_Player', 'Smithing_bot', 'Mining_bot', 
    'Magic_bot', 'PVM_Ranged_bot',  
    'Fletching_bot', 'PVM_Melee_bot', 'Herblore_bot',
    'Thieving_bot','Crafting_bot', 'PVM_Ranged_Magic_bot',
    'Hunter_bot','Runecrafting_bot','Fishing_bot','Agility_bot',
    'Cooking_bot', 'mort_myre_fungus_bot', 
    'Woodcutting_bot', 'Fishing_Cooking_bot',
    'Agility_Thieving_bot', 'Construction_Magic_bot','Construction_Prayer_bot',
    'Zalcano_bot'
]

ml = model(LABELS)

async def loop_request(url, json):
    i = 1
    data = []
    while True:
        url = f'{detector_api}/v1/player/bulk?token={token}&row_count=100000&page={i}'
        logging.debug(url)
        res = requests.post(url, json=json).json()
        data += res
        if len(res) == 0:
            break
        i += 1
    return data

@app.on_event('startup')
async def initial_task():
    loop = asyncio.get_running_loop()
    loop.create_task(get_player_hiscores())
    return

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/load")
async def train():
    if ml.model is None:
        ml.load()
    return {'ok': 'ok'}

@app.post("/train")
async def train():
    #TODO: token

    #TODO: get all labels & id's
    url = f'{detector_api}/v1/label?token={token}'
    logging.debug(url)
    data = requests.get(url).json()

    labels = [d for d in data if d['label'] in LABELS]
    label_ids = [l['id'] for l in labels]
    #TODO: get all labeled users from db
    label_input = {}
    label_input['label_id'] = label_ids
    
    data = loop_request(url, label_input)

    print(len(data))
    #TODO: get all hiscore data latest for those users from db
    return {'ok': 'ok'}

async def get_player_hiscores():
    logging.debug('getting data')
    url = f'{detector_api}/v1/prediction/data?token={token}'
    logging.debug(url)
    data = requests.get(url).json()

    # if there is no data wait and try to see if there is new data
    if len(data) == 0:
        logging.debug('no data to predict')
        await asyncio.sleep(600)
        return await get_player_hiscores()

    # clean & filter data
    data = data_class(data)
    print(data.filter_features(base=False, feature=True, ratio=False))

    # make predictions
    # post predictions
    return await get_player_hiscores()
