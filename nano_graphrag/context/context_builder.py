"""
上下文构建器

提供查询上下文的构建功能。
"""

import asyncio
from typing import Dict, Any, Optional
from ..base import BaseGraphStorage, BaseKVStorage, CommunitySchema, TextChunkSchema, QueryParam, BaseVectorStorage
from .._utils import logger, list_of_list_to_csv
from .entity_finder import _find_most_related_community_from_entities
from .community_finder import _find_most_related_text_unit_from_entities
from .relation_finder import _find_most_related_edges_from_entities


async def _build_local_query_context(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> Optional[str]:
    """
    构建局部查询上下文
    
    Args:
        query: 查询文本
        knowledge_graph_inst: 知识图谱实例
        entities_vdb: 实体向量数据库
        community_reports: 社区报告存储
        text_chunks_db: 文本块存储
        query_param: 查询参数
        
    Returns:
        构建的上下文字符串，如果失败则返回None
    """
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    
    # 获取节点数据，处理字典返回值
    node_data_dict = await knowledge_graph_inst.get_nodes_batch([r["entity_name"] for r in results])
    node_data_list = [node_data_dict.get(r["entity_name"]) for r in results]
    
    if not all([n is not None for n in node_data_list]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["entity_name"] for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_data_list, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
""" 