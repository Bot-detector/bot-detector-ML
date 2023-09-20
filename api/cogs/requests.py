import logging
import api.config as config
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


# Define an asynchronous function to make a secure HTTP GET request
async def make_request(url: str, params: dict, headers: dict = {}) -> list[dict]:
    # Create a secure copy of the parameters by adding a placeholder for the token
    _secure_params = params.copy()
    _secure_params["token"] = "***"

    # Log the URL and secure parameters for debugging
    logger.info({"url": url, "params": _secure_params})

    # Use aiohttp to make an asynchronous GET request
    async with aiohttp.ClientSession() as session:
        async with session.get(url=url, params=params, headers=headers) as resp:
            # Check if the response status is OK (200)
            if not resp.ok:
                error_message = (
                    f"response status {resp.status} "
                    f"response body: {await resp.text()}"
                )
                # Log the error message and raise a ValueError
                logger.error(error_message)
                raise ValueError(error_message)

            # Parse the response JSON and return the data
            data = await resp.json()
            return data


# Define an asynchronous function to retry a request until it succeeds or raise an exception on failure
async def retry_request(url: str, params: dict) -> list[dict]:
    while True:
        try:
            # Attempt to make the request
            data = await make_request(url, params)

            # If data is received, return it
            if data:
                return data
        except Exception as e:
            # Log the error and wait for 15 seconds before retrying
            _secure_params = params.copy()
            _secure_params["token"] = "***"
            logger.error({"url": url, "params": _secure_params, "error": str(e)})
            await asyncio.sleep(15)


# Define an asynchronous function to get labels from an API
async def get_labels():
    # Construct the URL and parameters for the request
    url = f"{config.detector_api}/v1/label"
    params = {
        "token": config.token,
    }

    # Retry the request until it succeeds and return the data
    data = await retry_request(url=url, params=params)
    return data


# Define an asynchronous function to get player data for a specific label ID
async def get_player_data(label_id):
    # Initialize pagination parameters
    page = 1
    row_count = 5000
    url = f"{config.detector_api}/v1/player/bulk"

    params = {
        "token": config.token,
        "page": page,
        "row_count": row_count,
        "label_id": label_id,
    }

    # Initialize a list to store player data
    players = []

    # Continue making requests until all data is retrieved
    while True:
        data = await retry_request(url=url, params=params)
        players.extend(data)

        logger.info(f"received: {len(data)}, in total {len(players)}")

        # Check if the received data is less than the row count, indicating the end of data
        if len(data) < row_count:
            break

        # Increment the page parameter for the next request
        params["page"] += 1

    return players


# Define an asynchronous function to get hiscore data for a specific label ID
async def get_hiscore_data(label_id):
    # Initialize pagination parameters
    page = 1
    row_count = 5000
    url = f"{config.detector_api}/v1/hiscore/Latest/bulk"

    params = {
        "token": config.token,
        "page": page,
        "row_count": row_count,
        "label_id": label_id,
    }

    # Initialize a list to store hiscore data
    hiscores = []

    # Continue making requests until all data is retrieved
    while True:
        data = await retry_request(url=url, params=params)
        hiscores.extend(data)

        logger.info(f"received: {len(data)}, in total {len(hiscores)}")

        # Check if the received data is less than the row count, indicating the end of data
        if len(data) < row_count:
            break

        # Increment the page parameter for the next request
        params["page"] += 1

    return hiscores


async def get_prediction_data():
    url = f"{config.detector_api}/v1/prediction/data"
    params = {"token": config.token, "limit": config.BATCH_AMOUNT}

    data = await retry_request(url=url, params=params)
    return data


async def post_prediction(data: list[dict]):
    url = f"{config.detector_api}/v1/prediction"
    params = {"token": config.token}

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url=url, params=params, data=data) as resp:
                    if not resp.ok:
                        continue
                    break
        except Exception as e:
            logger.error(str(e))
            await asyncio.sleep(60)
