import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

# load env variables
load_dotenv(find_dotenv(), verbose=True)

# get env variables
token = os.environ.get('token')
detector_api = os.environ.get('api')
secret_token = os.environ.get('secret_token')

app = FastAPI()

# setup logging
logger = logging.getLogger()
file_handler = logging.FileHandler(filename="./MachineLearning/models/error.log", mode='a')
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(filename='error.log', level=logging.DEBUG)

# log formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# add handler
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.error").propagate = False
