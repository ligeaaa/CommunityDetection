import asyncio
import functools
import threading
from PyQt5.QtCore import QThread
import time

import loguru


class AsyncQt5:
    def __init__(self, loop=None):
        self.loop = loop

    def get_loop(self, loop):
        self.loop = loop
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def __run(self, func, callback, errcall):
        try:
            if asyncio.iscoroutine(func):
                resp = await func
            else:
                resp = func()

            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(resp)
                else:
                    callback(resp)
        except Exception as e:
            if errcall:
                if asyncio.iscoroutinefunction(errcall):
                    await errcall(e)
                else:
                    errcall(e)

    def async_run(self, func, callback=None, errcall=None):
        try:
            new_loop = asyncio.new_event_loop()  # 在当前线程下创建时间循环，（未启用），在start_loop里面启动它  
            t = threading.Thread(target=self.get_loop, args=(new_loop,))  # 通过当前线程开启新的线程去启动事件循环  
            t.daemon = True
            t.start()
            future = asyncio.run_coroutine_threadsafe(self.__run(func, callback, errcall),
                                                      new_loop)  # 这几个是关键，代表在新线程中事件循环不断“游走”执行
            return future
        except Exception as e:
            loguru.logger.exception(e)
            raise e

    def __call__(self, *args, callback=None, **kwargs):
        def decorate(func):
            @functools.wraps(func)
            def warpper(*args, **kwargs):
                return self.async_run(func(*args, **kwargs), callback)

            return warpper

        return decorate


async_qt = AsyncQt5()

if __name__ == '__main__':

    def call(*args):
        print(args)


    @async_qt(callback=call)
    async def do(i):
        print(i)
        await asyncio.sleep(i)
        print(f"ddd {i}]")


    for i in range(100):
        do(i)

    time.sleep(10)