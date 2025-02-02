import asyncio
from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor


# executor = ProcessPoolExecutor(
#         max_workers=3,  
#     )  


async def run_in_loop(func, *args, executor=None, **kwargs):
    loop = asyncio.get_running_loop()
    f = partial(func, **kwargs)
    return await loop.run_in_executor(executor, f, *args)