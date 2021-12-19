import asyncio
import logging
import re
from typing import List

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

async def get_request(session:aiohttp.ClientSession, url:str) -> dict:
    pattern = r'token=.*&'
    clean_url = re.sub(pattern, 'token=***&', url)
    logger.debug(f'Request: url={clean_url}')
    async with session.get(url, ssl=False) as resp:
        if resp.status !=200:
            logger.debug(f'Break: {resp.status=}, {url=}')
            logger.debug(await resp.text())
            return None
        return await resp.json()

async def post_request(session:aiohttp.ClientSession, url:str, json:dict):
    async with session.post(url, json=json) as resp:
        pass

async def batch_request(urls:List, batch_size:int=5) -> List:
    data = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            tasks = []
            # create tasks
            for url in batch:
                future = asyncio.ensure_future(get_request(session, url))
                tasks.append(future)
            
            # await tasks
            outputs = await asyncio.gather(*tasks)
            
            for output in outputs:
                if output is None:
                    continue
                data.extend(output)
            logger.debug(f'batch: {i/batch_size}, outputs: {np.shape(outputs)}, data: {np.shape(data)}')
    return data

#TODO: seperate file for helper functions
async def loop_request(base_url:str) -> List[dict]:
    '''
        this function gets all the data of a paginated route
    '''
    i, data = 1, []
    
    async with aiohttp.ClientSession() as session:
        while True:
            # build url
            url = f'{base_url}&row_count=100000&page={i}'

            # logging
            logger.debug(f'Request: {url=}')

            # make request
            response = await get_request(session, url)
            
            # escape condition
            if response is None:
                break

            # parse data
            data += response

            # escape condition
            if len(response) == 0:
                logger.debug(f'Break: {len(response)=}, {url=}')
                break
            
            # logging (after break)
            logger.debug(f'Succes: {len(response)=}, {len(data)=} {i=}')

            # update iterator
            i += 1
    return data
