import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

# TODO: make this an async function
def request(urls: List) -> List[dict]:
    output = []
    for url in urls:
        logger.debug(url)
        data = requests.get(url).json()
        output.extend(data)
    return output
