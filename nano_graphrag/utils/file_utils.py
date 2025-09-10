"""
文件操作工具模块

本模块提供文件操作相关的工具函数，包括：
- JSON文件读写
- 时间戳生成
- 文件路径管理
"""

import os
import json
import time
import logging
from typing import Any

# 创建日志记录器
logger = logging.getLogger("nano-graphrag")


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        时间戳字符串，格式为'YYYYMMDD_HHMMSS'
        
    Example:
        >>> get_timestamp()
        '20241201_143052'
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
        
    Returns:
        是否保存成功
        
    Example:
        >>> save_json({'key': 'value'}, 'data.json')
        True
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            
        return True
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        return False


def load_json(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON数据，如果失败返回None
        
    Example:
        >>> load_json('data.json')
        {'key': 'value'}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        return None


def write_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    写入JSON文件（save_json的别名）
    
    Args:
        data: 要写入的数据
        file_path: 文件路径
        indent: 缩进
        
    Returns:
        是否成功
        
    Example:
        >>> write_json({'key': 'value'}, 'data.json')
        True
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        logger.error(f"写入JSON文件失败: {e}")
        return False 