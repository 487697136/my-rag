"""
全局图检索查询

提供基于全局图结构的查询功能。
"""

import asyncio
import tiktoken
from typing import Dict, Any, List, Union
from ..base import BaseGraphStorage, BaseKVStorage, CommunitySchema, TextChunkSchema, QueryParam, BaseVectorStorage
from .._utils import truncate_list_by_token_size, logger, list_of_list_to_csv
from ..answer_generation.prompts import PROMPTS
from ..retrieval.alignment import RetrievalResult, create_retrieval_adapter


async def _map_global_communities(
    query: str,
    communities_data: List[CommunitySchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """
    映射全局社区
    
    Args:
        query: 查询文本
        communities_data: 社区数据列表
        query_param: 查询参数
        global_config: 全局配置
        
    Returns:
        映射结果列表
    """
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        # 创建tiktoken编码器
        tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
        this_group = truncate_list_by_token_size(
            communities_data,
            query_param.global_max_token_for_community_report,
            tiktoken_model,
            key=lambda x: x["report_string"]
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: List[CommunitySchema]) -> Dict[str, Any]:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        # 添加调试信息
        logger.debug(f"LLM response type: {type(response)}, content: {response[:200] if isinstance(response, str) else response}")
        data = use_string_json_convert_func(response)
        logger.debug(f"Converted data type: {type(data)}, content: {data}")
        if not isinstance(data, dict):
            logger.error(f"Expected dict but got {type(data)}: {data}")
            return []
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
    return_context: bool = False,
    return_raw_results: bool = False
) -> Union[str, List[RetrievalResult]]:
    """
    全局图检索查询
    
    Args:
        query: 查询文本
        knowledge_graph_inst: 知识图谱实例
        entities_vdb: 实体向量数据库
        community_reports: 社区报告存储
        text_chunks_db: 文本块存储
        query_param: 查询参数
        global_config: 全局配置
        return_context: 是否只返回上下文
        return_raw_results: 是否返回原始结果列表(List[RetrievalResult])
        
    Returns:
        回答、上下文字符串或原始结果列表
    """
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Retrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        if return_raw_results:
            return []
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        query_param.global_max_token_for_community_report,
        tiktoken_model,
        key=lambda x: x["answer"]
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    
    # 如果需要返回原始结果，使用适配器转换
    if return_raw_results:
        try:
            adapter = create_retrieval_adapter()
            retrieval_results = await adapter.adapt_graph_results(points_context, query, source="global")
            return retrieval_results
        except Exception as adapter_error:
            logger.error(f"Global适配器转换失败: {adapter_error}")
            return []
    
    if query_param.only_need_context or return_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response 