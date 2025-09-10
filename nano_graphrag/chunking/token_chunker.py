"""
基于token大小的文档分块器

提供基于token数量进行文档分块的功能，支持重叠分块。
"""

import tiktoken
from typing import List, Dict, Any


def chunking_by_token_size(
    tokens_list: List[List[int]],
    doc_keys: List[str],
    tiktoken_model: tiktoken.Encoding,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> List[Dict[str, Any]]:
    """
    基于token大小进行文档分块
    
    Args:
        tokens_list: 文档的token列表，每个元素是一个文档的token序列
        doc_keys: 文档ID列表，与tokens_list对应
        tiktoken_model: tiktoken编码器实例
        overlap_token_size: 重叠token数量，默认为128
        max_token_size: 最大token数量，默认为1024
        
    Returns:
        分块结果列表，每个元素包含tokens、content、chunk_order_index、full_doc_id
    """
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # 解码token为文本
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results 