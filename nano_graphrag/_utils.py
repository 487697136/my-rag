"""
工具函数模块 - 已重构

本模块已重构为多个专门的模块：
- utils.cache_utils: 缓存管理功能
- utils.text_processing: 文本处理功能
- utils.file_utils: 文件操作功能
- utils.async_utils: 异步操作功能
- utils.decorators: 装饰器功能

为了向后兼容，保留所有原有函数的导入。
"""

# 类型定义
from typing import Dict, List, Any, Optional, Union, Callable
EmbeddingFunc = Callable[[List[str]], List[List[float]]]

# 配置日志
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建日志记录器
logger = logging.getLogger("nano-graphrag")


# 从重构后的模块导入所有函数
from nano_graphrag.utils import (
    # 缓存管理
    get_cache_dir,
    clear_cache,
    compute_args_hash,
    compute_mdhash_id,
    
    # 文本处理
    normalize_text,
    compute_text_hash,
    split_text_by_length,
    clean_str,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    convert_response_to_json,
    
    # 文件操作
    get_timestamp,
    save_json,
    load_json,
    write_json,
    
    # 异步操作
    limit_async_func_call,
    always_get_an_event_loop,
    
    # 装饰器
    timer,
    wrap_embedding_func_with_attrs
)

# 导出所有函数以保持向后兼容
__all__ = [
    # 类型定义
    'EmbeddingFunc',
    
    # 缓存管理
    'get_cache_dir',
    'clear_cache',
    'compute_args_hash',
    'compute_mdhash_id',
    
    # 文本处理
    'normalize_text',
    'compute_text_hash',
    'split_text_by_length',
    'clean_str',
    'decode_tokens_by_tiktoken',
    'encode_string_by_tiktoken',
    'is_float_regex',
    'list_of_list_to_csv',
    'pack_user_ass_to_openai_messages',
    'split_string_by_multi_markers',
    'truncate_list_by_token_size',
    'convert_response_to_json',
    
    # 文件操作
    'get_timestamp',
    'save_json',
    'load_json',
    'write_json',
    
    # 异步操作
    'limit_async_func_call',
    'always_get_an_event_loop',
    
    # 装饰器
    'timer',
    'wrap_embedding_func_with_attrs'
]
