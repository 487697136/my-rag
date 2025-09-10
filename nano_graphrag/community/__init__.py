"""
社区检测和报告生成模块

提供图社区的检测和报告功能：
- 社区检测算法
- 社区报告生成
- 社区分析工具
"""

# 社区报告生成
from .report_generator import generate_community_report

# 社区打包和处理
from .community_packer import (
    _pack_single_community_by_sub_communities,
    _pack_single_community_describe,
)

__all__ = [
    # 主要函数
    "generate_community_report",
    
    # 内部函数
    "_pack_single_community_by_sub_communities",
    "_pack_single_community_describe",
] 