"""
分块管理器

提供文档分块的高级管理功能，包括分块函数的选择和调用。
"""

import tiktoken
from typing import Dict, Any, Callable
from .._utils import compute_mdhash_id


def get_chunks(
    new_docs: Dict[str, Dict[str, Any]], 
    chunk_func: Callable = None, 
    **chunk_func_params: Any
) -> Dict[str, Dict[str, Any]]:
    """
    获取文档分块
    
    Args:
        new_docs: 新文档字典，格式为 {doc_id: {"content": "文档内容"}}
        chunk_func: 分块函数，默认为None（使用默认的token分块）
        **chunk_func_params: 分块函数的额外参数
        
    Returns:
        分块结果字典，格式为 {chunk_id: chunk_data}
    """
    from .token_chunker import chunking_by_token_size
    
    if chunk_func is None:
        chunk_func = chunking_by_token_size
        
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks 