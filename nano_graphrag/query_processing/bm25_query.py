"""
BM25检索查询

提供基于BM25算法的文档检索功能。
"""

import tiktoken
from typing import Dict, Any, List, Union
from ..base import BaseKVStorage, TextChunkSchema, QueryParam
from .._utils import truncate_list_by_token_size, logger
from ..answer_generation.prompts import PROMPTS
from ..retrieval.alignment import RetrievalResult, create_retrieval_adapter


async def bm25_query(
    query: str,
    bm25_storage: Any,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
    return_raw_results: bool = False
) -> Union[str, List[RetrievalResult]]:
    """
    BM25检索查询
    
    Args:
        query: 查询文本
        bm25_storage: BM25存储实例
        text_chunks_db: 文本块键值存储
        query_param: 查询参数
        global_config: 全局配置
        return_raw_results: 是否返回原始结果列表(List[RetrievalResult])
        
    Returns:
        回答、上下文字符串或原始结果列表
    """
    use_model_func = global_config["best_model_func"]

    # 兼容：若未启用或未初始化 BM25，直接返回失败响应
    if bm25_storage is None:
        logger.warning("BM25 storage not available; please enable_bm25 and build index.")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]

    # 通过BM25检索相关文档
    results = await bm25_storage.search(query, top_k=query_param.top_k)
    
    if not len(results):
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
    
    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    maybe_trun_chunks = truncate_list_by_token_size(
        results,
        query_param.bm25_max_token_for_text_unit,
        tiktoken_model,
        key=lambda x: x["content"]
    )
    logger.info(f"BM25搜索 - 截断 {len(results)} 到 {len(maybe_trun_chunks)} chunks")
    
    # 如果需要返回原始结果，使用适配器转换
    if return_raw_results:
        try:
            adapter = create_retrieval_adapter()
            retrieval_results = await adapter.adapt_bm25_results(maybe_trun_chunks, query)
            return retrieval_results
        except Exception as adapter_error:
            logger.error(f"BM25适配器转换失败: {adapter_error}")
            return []
    
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    
    if query_param.only_need_context:
        return section
    
    # 使用LLM生成回答
    sys_prompt_temp = PROMPTS["naive_rag_response"]  # 复用naive模式的提示模板
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response 