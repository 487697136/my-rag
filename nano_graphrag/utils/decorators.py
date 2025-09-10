"""
装饰器工具模块

本模块提供各种装饰器工具函数，包括：
- 函数执行计时
- 嵌入函数属性包装
- 其他通用装饰器
"""

import time
import logging
from typing import Callable, Any

# 创建日志记录器
logger = logging.getLogger("nano-graphrag")


def timer(func: Callable) -> Callable:
    """
    函数执行计时装饰器
    
    Args:
        func: 要计时的函数
        
    Returns:
        包装后的函数
        
    Example:
        @timer
        def my_function():
            # 函数执行时会自动记录耗时
            pass
    """
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f}秒")
        return result
    return wrapper


def wrap_embedding_func_with_attrs(embedding_dim: int = 1536, max_token_size: int = 8192):
    """
    为嵌入函数添加属性的装饰器
    
    Args:
        embedding_dim: 嵌入维度
        max_token_size: 最大token数量
        
    Returns:
        装饰器函数
        
    Example:
        @wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=4096)
        def my_embedding_function(texts):
            # 函数会自动获得 embedding_dim 和 max_token_size 属性
            pass
    """
    def decorator(func: Callable) -> Callable:
        func.embedding_dim = embedding_dim
        func.max_token_size = max_token_size
        return func
    return decorator 