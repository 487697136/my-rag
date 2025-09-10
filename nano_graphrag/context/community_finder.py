"""
社区查找器

提供基于实体的文本单元查找功能。
"""

import tiktoken
from typing import List, Dict, Any
from ..base import BaseKVStorage, TextChunkSchema, BaseGraphStorage, QueryParam
from .._utils import truncate_list_by_token_size, logger, split_string_by_multi_markers
from ..answer_generation.prompts import GRAPH_FIELD_SEP


async def _find_most_related_text_unit_from_entities(
    node_datas: List[Dict[str, Any]],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
) -> List[TextChunkSchema]:
    """
    从实体中查找最相关的文本单元
    
    Args:
        node_datas: 节点数据列表
        query_param: 查询参数
        text_chunks_db: 文本块存储
        knowledge_graph_inst: 知识图谱实例
        
    Returns:
        相关文本单元列表
    """
    import asyncio
    
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data_dict = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in all_one_hop_nodes_data_dict.items()
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        query_param.local_max_token_for_text_unit,
        tiktoken_model,
        key=lambda x: x["data"]["content"]
    )
    all_text_units: List[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units 