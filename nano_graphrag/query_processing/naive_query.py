"""
朴素向量检索查询

提供基于向量相似度的文档检索功能。
"""

import tiktoken
from typing import Dict, Any, List, Union
from ..base import BaseVectorStorage, BaseKVStorage, TextChunkSchema, QueryParam
from .._utils import truncate_list_by_token_size, logger
from ..answer_generation.prompts import PROMPTS
from ..retrieval.alignment import RetrievalResult, create_retrieval_adapter


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
    return_context: bool = False,
    return_raw_results: bool = False
) -> Union[str, List[RetrievalResult]]:
    """
    朴素向量检索查询
    
    Args:
        query: 查询文本
        chunks_vdb: 文本块向量数据库
        text_chunks_db: 文本块键值存储
        query_param: 查询参数
        global_config: 全局配置
        return_context: 是否只返回上下文
        return_raw_results: 是否返回原始结果列表(List[RetrievalResult])
        
    Returns:
        回答、上下文字符串或原始结果列表
    """
    use_model_func = global_config["best_model_func"]
    try:
        results = await chunks_vdb.query(query, top_k=query_param.top_k)
    except Exception as e:
        logger.error(f"向量查询失败: {e}")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
        
    if not len(results):
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
    
    chunks_ids = [r["id"] for r in results]
    try:
        chunks = await text_chunks_db.get_by_ids(chunks_ids)
    except Exception as e:
        logger.error(f"文本块检索失败: {e}, chunks_ids: {chunks_ids}")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
        
    # 过滤掉None值以避免'NoneType' object is not subscriptable错误
    chunks = [c for c in chunks if c is not None and isinstance(c, dict)]
    
    # 如果所有chunks都是None，尝试重新查询或返回失败响应
    if not chunks:
        logger.warning(f"No valid chunks found for query: {query}")
        # 尝试降低查询要求
        try:
            if query_param.top_k > 5:
                logger.info("尝试降低top_k重新查询")
                results = await chunks_vdb.query(query, top_k=5)
                if results:
                    chunks_ids = [r["id"] for r in results]
                    chunks = await text_chunks_db.get_by_ids(chunks_ids)
                    chunks = [c for c in chunks if c is not None and isinstance(c, dict)]
        except Exception as retry_error:
            logger.warning(f"重试查询失败: {retry_error}")
        
        if not chunks:
            if return_raw_results:
                return []
            return PROMPTS["fail_response"]

    # 验证chunks内容
    valid_chunks = []
    for chunk in chunks:
        if chunk and isinstance(chunk, dict) and "content" in chunk:
            content = chunk.get("content", "")
            if content and isinstance(content, str) and len(content.strip()) > 0:
                valid_chunks.append(chunk)
    
    if not valid_chunks:
        logger.warning(f"No chunks with valid content found for query: {query}")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]

    # 创建tiktoken编码器
    try:
        tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    except Exception as tiktoken_error:
        logger.warning(f"Tiktoken模型加载失败: {tiktoken_error}, 使用默认编码")
        tiktoken_model = tiktoken.get_encoding("cl100k_base")
        
    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        query_param.naive_max_token_for_text_unit,
        tiktoken_model,
        key=lambda x: x.get("content", "")
    )
    logger.info(f"Truncate {len(valid_chunks)} to {len(maybe_trun_chunks)} chunks")
    
    if not maybe_trun_chunks:
        logger.warning(f"No chunks remaining after truncation for query: {query}")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
        
    # 安全地连接内容
    try:
        content_parts = []
        for c in maybe_trun_chunks:
            content = c.get("content", "")
            if content and isinstance(content, str):
                content_parts.append(content.strip())
        
        if not content_parts:
            logger.warning(f"No valid content parts found for query: {query}")
            if return_raw_results:
                return []
            return PROMPTS["fail_response"]
            
        section = "--New Chunk--\n".join(content_parts)
    except Exception as content_error:
        logger.error(f"内容拼接失败: {content_error}")
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
    
    # 如果需要返回原始结果，使用适配器转换
    if return_raw_results:
        try:
            adapter = create_retrieval_adapter()
            # 构建原始结果格式，合并results和chunks信息
            raw_results_for_adapter = []
            for i, result in enumerate(results):
                if i < len(maybe_trun_chunks):
                    chunk = maybe_trun_chunks[i]
                    raw_results_for_adapter.append({
                        "id": result.get("id"),
                        "score": result.get("score", 0.0),
                        "content": chunk.get("content", "")
                    })
            
            retrieval_results = await adapter.adapt_naive_results(raw_results_for_adapter, query)
            return retrieval_results
        except Exception as adapter_error:
            logger.error(f"适配器转换失败: {adapter_error}")
            return []
    
    if query_param.only_need_context or return_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    try:
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )
        
        # 验证响应类型和内容
        if not isinstance(response, str):
            logger.error(f"LLM返回非字符串响应: type={type(response)}, value={response}")
            response = str(response)
        
        if len(response.strip()) == 0:
            logger.warning(f"LLM返回空响应")
            return PROMPTS["fail_response"]
        
        # 检查是否返回了数字（这是一个异常情况）
        if response.strip().isdigit():
            logger.error(f"LLM返回纯数字响应(异常): {response}")
            return PROMPTS["fail_response"]
        
        logger.debug(f"LLM响应正常: 长度={len(response)}, 前50字符={response[:50]}")
        return response
        
    except Exception as llm_error:
        logger.error(f"LLM调用失败: {llm_error}")
        # 注意：这里不应该返回空列表，因为到这里时return_raw_results应该已经被处理了
        return PROMPTS["fail_response"] 