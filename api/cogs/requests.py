import logging
from typing import List

from urllib3 import Retry
import api.config as config
import requests
import time

logger = logging.getLogger(__name__)

def request(
    urls: List[str], retry: int = 3, sleep: int = 10
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

        resp = requests.get(url)
        retrying = False
        if resp.ok:
            try:
                data = resp.json()
            except Exception as e:
                logger.exception(e)
                retrying = True
        else:
            logger.error(f"received status: {resp.status_code}")
            retrying = True

        if retrying:
            if retry > 0:
                time.sleep(sleep)
                data = request([url], retry - 1, sleep)
            else:
                data = []

        output.extend(data)
        logger.debug(f"Output size: {len(data)}")
    return output