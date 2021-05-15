import requests
import pandas as pd


def get_highscores(token):
    url = f'http://45.33.127.106:5000/site/highscores/{token}/1'
    response = requests.get(url)
    return pd.DataFrame(response.json())

def get_players(token, ofintrest=True):
    url = f'http://45.33.127.106:5000/site/players/{token}/1'
    response = requests.get(url)
    return pd.DataFrame(response.json())

def get_labels(token):
    url = f'http://45.33.127.106:5000/site/labels/{token}'
    response = requests.get(url)
    return pd.DataFrame(response.json())