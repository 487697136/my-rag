"""
纯LLM查询

提供不进行任何检索，直接使用LLM生成回答的功能。
"""

from typing import Dict, Any
from ..base import QueryParam


async def llm_only_query(
    query: str,
    query_param: QueryParam,
    global_config: Dict[str, Any],
) -> str:
    """
    纯LLM查询
    
    Args:
        query: 查询文本
        query_param: 查询参数
        global_config: 全局配置
        
    Returns:
        LLM生成的回答
    """
    use_model_func = global_config["best_model_func"]
    
    # 直接使用LLM生成回答，不进行任何检索
    response = await use_model_func(
        query,
        system_prompt=query_param.llm_only_system_prompt,
    )
    return response 