import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

# TODO: make this an async function
def request(urls: List) -> List[dict]:
    output = []
    for url in urls:
        logger.debug(url)
        try:
            data = requests.get(url)
            data = data.json()
        except:
            print(data.text())
            data = request([url])
        output.extend(data)
    return output
