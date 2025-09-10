"""
异步操作工具模块

本模块提供异步操作相关的工具函数，包括：
- 异步函数并发限制
- 事件循环管理
- 异步装饰器
"""

import asyncio
from functools import wraps
from typing import Callable, Any


def limit_async_func_call(max_concurrent: int):
    """
    限制异步函数并发调用的装饰器
    
    Args:
        max_concurrent: 最大并发数
        
    Returns:
        装饰器函数
        
    Example:
        @limit_async_func_call(5)
        async def my_async_function():
            # 最多5个并发调用
            pass
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def always_get_an_event_loop():
    """
    获取或创建事件循环
    
    Returns:
        事件循环
        
    Example:
        >>> loop = always_get_an_event_loop()
        >>> asyncio.run_coroutine_threadsafe(coro, loop)
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop() 