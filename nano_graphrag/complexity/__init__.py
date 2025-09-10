"""
复杂度分类与路由模块

提供基于ModernBERT的查询复杂度分类和智能路由功能。
"""

# 导入复杂度分类器
from .classifier import (
    ComplexityClassifier,
    ComplexityClassifierConfig,
    get_global_classifier,
    classify_query_complexity,
    classify_query_complexity_sync,
)

# 导入复杂度感知路由器
from .router import (
    BaseRouter,
    ComplexityAwareRouter,
)

# 导出所有函数以保持向后兼容
__all__ = [
    # 复杂度分类器
    'ComplexityClassifier',
    'ComplexityClassifierConfig',
    'get_global_classifier',
    'classify_query_complexity',
    'classify_query_complexity_sync',
    # 复杂度感知路由器
    'BaseRouter',
    'ComplexityAwareRouter',
] 