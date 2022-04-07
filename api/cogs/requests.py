import logging
from typing import List
import api.config as config
import requests
import time

logger = logging.getLogger(__name__)

def request(
    urls: List[str], retry: int = 3, sleep: int = 1
) -> List[dict]:
    """
    request data from urls.

    :param urls: list of urls
    :param retry: number of retries
    :param sleep: number of seconds to sleep
    :return: list of dicts.
    """
    output = []
    for url in urls:
        logger.debug(url.replace(config.token, "***"))
        try:
            data = requests.get(url)
            data = data.json()
        except Exception as e:
            logger.exception(e)
            if retry > 0:
                time.sleep(sleep)
                data = request([url], retry - 1, sleep)
            else:
                data = []
        output.extend(data)
    return output