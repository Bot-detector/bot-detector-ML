from typing import List

import pandas as pd
from fastapi import FastAPI

from classes.Player import Player
from model import model as Model

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predictions")
def predict(players: List[Player]):
    data = Model.data(pd.DataFrame(players))
    prediction = model.predict(data)
    return prediction

@app.post("/train-model")
def train():
    global model
    # get training data
    df = pd.DataFrame()
    model = Model.model(df)
    model.train()
    return {'OK':'OK'}
# run on init    
train()