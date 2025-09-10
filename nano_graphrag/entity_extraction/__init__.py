"""
实体提取模块

基于DSPy框架的实体和关系提取，支持：
- 实体识别和分类
- 关系抽取
- 图构建
- 评估指标
"""

# 主要提取函数
from .extract import (
    extract_entities,  # 主要的实体提取函数
    generate_dataset,
    extract_entities_dspy,
)

# 实体和关系模型
from .module import (
    Entity,
    Relationship, 
    CombinedExtraction,
    CritiqueCombinedExtraction,
    RefineCombinedExtraction,
    TypedEntityRelationshipExtractor,
    TypedEntityRelationshipExtractorException,
)

# 评估指标
from .metric import (
    relationships_similarity_metric,
    entity_recall_metric,
    AssessRelationships,
)

__all__ = [
    # 主要函数
    "extract_entities",
    "generate_dataset", 
    "extract_entities_dspy",
    
    # 数据模型
    "Entity",
    "Relationship",
    "CombinedExtraction",
    "CritiqueCombinedExtraction", 
    "RefineCombinedExtraction",
    "TypedEntityRelationshipExtractor",
    "TypedEntityRelationshipExtractorException",
    
    # 评估指标
    "relationships_similarity_metric",
    "entity_recall_metric", 
    "AssessRelationships",
]
