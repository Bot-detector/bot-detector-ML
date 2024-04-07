import json
import logging
import sys

from src.core.config import settings

# # log formatting
formatter = logging.Formatter(
    json.dumps(
        {
            "ts": "%(asctime)s",
            "name": "%(name)s",
            "function": "%(funcName)s",
            "level": "%(levelname)s",
            "msg": json.dumps("%(message)s"),
        }
    )
)

stream_handler = logging.StreamHandler(sys.stdout)

stream_handler.setFormatter(formatter)

handlers = [stream_handler]

logging.basicConfig(level=logging.DEBUG, handlers=handlers)

if settings.ENV == "PRD":
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.disabled = True
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
