import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), verbose=True)

token = os.environ['token']
