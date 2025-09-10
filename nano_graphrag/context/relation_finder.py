"""
关系查找器

提供基于实体的关系查找功能。
"""

import tiktoken
from typing import List, Dict, Any
from ..base import BaseGraphStorage, QueryParam
from .._utils import truncate_list_by_token_size, logger


async def _find_most_related_edges_from_entities(
    node_datas: List[Dict[str, Any]],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
) -> List[Dict[str, Any]]:
    """
    从实体中查找最相关的关系
    
    Args:
        node_datas: 节点数据列表
        query_param: 查询参数
        knowledge_graph_inst: 知识图谱实例
        
    Returns:
        相关关系列表
    """
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    
    all_edges = []
    seen = set()
    
    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge) 
                
    all_edges_pack = await knowledge_graph_inst.get_edges_batch(all_edges)
    all_edges_degree = await knowledge_graph_inst.edge_degrees_batch(all_edges)
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        query_param.local_max_token_for_local_context,
        tiktoken_model,
        key=lambda x: x["description"]
    )
    return all_edges_data 