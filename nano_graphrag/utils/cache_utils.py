"""
缓存管理工具模块

本模块提供缓存相关的工具函数，包括：
- 缓存目录管理
- 缓存清理
- 参数哈希计算
- 文本哈希ID生成
"""

import os
import json
import hashlib
import logging
from typing import Optional

# 创建日志记录器
logger = logging.getLogger("nano-graphrag")


def get_cache_dir(cache_name: Optional[str] = None) -> str:
    """
    获取缓存目录路径
    
    Args:
        cache_name: 缓存名称，默认为None
        
    Returns:
        缓存目录路径
        
    Example:
        >>> get_cache_dir()
        '/path/to/nano_graphrag_cache'
        >>> get_cache_dir('my_cache')
        '/path/to/nano_graphrag_cache/my_cache'
    """
    # 默认缓存目录
    base_cache_dir = os.environ.get(
        "NANO_GRAPHRAG_CACHE_DIR", 
        os.path.join(os.getcwd(), "nano_graphrag_cache")
    )
    
    # 如果指定了缓存名称，则添加到路径中
    if cache_name:
        cache_dir = os.path.join(base_cache_dir, cache_name)
    else:
        cache_dir = base_cache_dir
        
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir


def clear_cache(cache_name: Optional[str] = None) -> bool:
    """
    清除缓存
    
    Args:
        cache_name: 缓存名称，默认为None表示清除所有缓存
        
    Returns:
        是否成功清除
        
    Example:
        >>> clear_cache()  # 清除所有缓存
        True
        >>> clear_cache('my_cache')  # 清除特定缓存
        True
    """
    import shutil
    
    try:
        if cache_name:
            # 清除特定缓存
            cache_dir = get_cache_dir(cache_name)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.info(f"已清除缓存: {cache_dir}")
        else:
            # 清除所有缓存
            base_cache_dir = os.environ.get(
                "NANO_GRAPHRAG_CACHE_DIR", 
                os.path.join(os.getcwd(), "nano_graphrag_cache")
            )
            if os.path.exists(base_cache_dir):
                shutil.rmtree(base_cache_dir)
                logger.info(f"已清除所有缓存: {base_cache_dir}")
                
        return True
    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        return False


def compute_args_hash(*args, **kwargs) -> str:
    """
    计算参数的哈希值，用于缓存
    
    Args:
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        参数的哈希值
        
    Example:
        >>> compute_args_hash('hello', world=123)
        'a1b2c3d4e5f6...'
    """
    # 将参数转换为可序列化的格式
    args_str = json.dumps([args, kwargs], sort_keys=True, default=str)
    return hashlib.md5(args_str.encode()).hexdigest()


def compute_mdhash_id(text: str, prefix: str = "") -> str:
    """
    计算文本的MD5哈希ID
    
    Args:
        text: 输入文本
        prefix: ID前缀
        
    Returns:
        哈希ID
        
    Example:
        >>> compute_mdhash_id("hello world")
        '5eb63bbbe01eeed093cb22bb8f5acdc3'
        >>> compute_mdhash_id("hello world", "doc_")
        'doc_5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"{prefix}{hash_value}" if prefix else hash_value 