"""
Retrieval Layer Module
提供统一的检索结果对齐和适配功能
"""

from .alignment import (
    RetrievalResult,
    ScoreNormalizer,
    RetrievalResultAdapter
)

__all__ = [
    "RetrievalResult",
    "ScoreNormalizer", 
    "RetrievalResultAdapter"
]
