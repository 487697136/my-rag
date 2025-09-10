"""
工具函数包 - 已重构

本包包含重构后的工具函数模块：
- cache_utils: 缓存管理相关函数
- text_processing: 文本处理相关函数
- file_utils: 文件操作相关函数
- async_utils: 异步操作相关函数
- decorators: 装饰器相关函数

为了向后兼容，保留所有原有函数的导入。
"""

# 缓存管理功能
from .cache_utils import (
    get_cache_dir,
    clear_cache,
    compute_args_hash,
    compute_mdhash_id
)

# 文本处理功能
from .text_processing import (
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
    convert_response_to_json
)

# 文件操作功能
from .file_utils import (
    get_timestamp,
    save_json,
    load_json,
    write_json
)

# 异步操作功能
from .async_utils import (
    limit_async_func_call,
    always_get_an_event_loop
)

# 装饰器功能
from .decorators import (
    timer,
    wrap_embedding_func_with_attrs
)

# 导出所有函数以保持向后兼容
__all__ = [
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