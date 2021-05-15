from typing import Optional
from fastapi import FastAPI
from classes.Player import Player

app = FastAPI()
db =[]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predictions")
def get_prediction(player: Player):
    print(player)
    db.append(player.dict())
    return db[-1]