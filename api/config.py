import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

# load env variables
load_dotenv(find_dotenv(), verbose=True)

# get env variables
token = os.environ.get("token")
detector_api = os.environ.get("api")
secret_token = os.environ.get("secret_token")

app = FastAPI()

# setup logging
logger = logging.getLogger()
file_handler = logging.FileHandler(filename="error.log", mode="a")
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(filename="error.log", level=logging.DEBUG)

# log formatting
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# add handler
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.error").propagate = False

BATCH_AMOUNT = 5_000

LABELS = [
    "Real_Player",
    "PVM_Melee_bot",
    "Smithing_bot",
    "Magic_bot",
    "Fishing_bot",
    "Mining_bot",
    "Crafting_bot",
    "PVM_Ranged_Magic_bot",
    "PVM_Ranged_bot",
    "Hunter_bot",
    "Fletching_bot",
    "LMS_bot",
    "Agility_bot",
    "Wintertodt_bot",
    "Runecrafting_bot",
    "Zalcano_bot",
    "Woodcutting_bot",
    "Thieving_bot",
    "Soul_Wars_bot",
    "Cooking_bot",
    "Vorkath_bot",
    "Barrows_bot",
    "Herblore_bot",
    "Zulrah_bot",
    "Unknown_bot"
]
