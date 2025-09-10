"""
实体查找器

提供基于实体的社区查找功能。
"""

import json
import tiktoken
from collections import Counter
from typing import List, Dict, Any
from ..base import BaseKVStorage, CommunitySchema, QueryParam
from .._utils import truncate_list_by_token_size, logger


async def _find_most_related_community_from_entities(
    node_datas: List[Dict[str, Any]],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
) -> List[Dict[str, Any]]:
    """
    从实体中查找最相关的社区
    
    Args:
        node_datas: 节点数据列表
        query_param: 查询参数
        community_reports: 社区报告存储
        
    Returns:
        相关社区列表
    """
    import asyncio
    
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        query_param.local_max_token_for_community_report,
        tiktoken_model,
        key=lambda x: x["report_string"]
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports 