"""
文本分块模块

提供多种文本分块策略：
- 基于token大小的分块
- 基于分隔符的分块  
- 动态分块管理
- 分块器工具
"""

# 主要分块函数
from .chunk_manager import get_chunks

# 分块策略
from .token_chunker import chunking_by_token_size
from .separator_chunker import chunking_by_seperators

# 分块器工具
from .splitter import SeparatorSplitter

__all__ = [
    # 主要函数
    "get_chunks",
    
    # 分块策略
    "chunking_by_token_size",
    "chunking_by_seperators",
    
    # 工具类
    "SeparatorSplitter",
] 