#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: decorator.py
@date: 2024/10/4 23:03
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

import asyncio
import time
from functools import wraps


def time_record(original_function):
    """
    用于获取目标函数名称与入参与其执行时间的装饰器类
    在执行方法后，在控制台输出类似 target_function(3, '测试', key='值') took 3.0009 seconds to run, from 1715336837.8986 to 1715336840.8995 的消息
    """

    @wraps(original_function)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = original_function(*args, **kwargs)
        end = time.time()
        print(f"took {end - start:.4f} seconds to run, from {start:.4f} to {end:.4f}")
        return result

    @wraps(original_function)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await original_function(*args, **kwargs)
        end = time.time()
        print(f"took {end - start:.4f} seconds to run, from {start:.4f} to {end:.4f}")
        return result

    if asyncio.iscoroutinefunction(original_function):
        return async_wrapper
    else:
        return sync_wrapper
