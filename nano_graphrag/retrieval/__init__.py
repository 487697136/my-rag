"""
检索模块

提供多源检索结果的统一表示、对齐和融合功能：
- RetrievalResult：统一的检索结果数据结构
- 结果对齐：将不同检索源的结果转换为统一格式
- RRF融合：基于互惠排名融合的多源结果合并
- 置信度感知：根据查询复杂度动态调整融合权重
"""

from .alignment import RetrievalResult, RetrievalAdapter, create_retrieval_adapter, align_retrieval_results
from .fusion import ConfidenceAwareFusion, FusionConfig, create_fusion_engine

__all__ = [
    # 数据结构
    "RetrievalResult",
    
    # 对齐功能
    "RetrievalAdapter",
    "create_retrieval_adapter",
    "align_retrieval_results",
    
    # 融合功能
    "ConfidenceAwareFusion",
    "FusionConfig", 
    "create_fusion_engine",
]