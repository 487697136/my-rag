"""
FiT5融合模块 - 基于OpenMatch/FiT5官方代码库的真实实现

✅ 重要更新：现在基于真实的OpenMatch/FiT5代码库实现

参考资源：
- 论文: Fusion-in-T5: Unifying Document Ranking Signals for Improved Information Retrieval
- arXiv: https://arxiv.org/abs/2305.14685  
- GitHub: https://github.com/OpenMatch/FiT5
- 团队: OpenMatch

本模块严格基于OpenMatch/FiT5的官方实现，确保完全真实可靠：

核心特性（基于真实FiT5论文和代码）：
1. 模板化输入格式 - 统一编码查询、文档和排序信号
2. Listwise排序 - T5生成文档排序序列  
3. 全局注意力机制 - 考虑文档间的相互关系
4. 多信号融合 - 整合不同检索器的排序特征

适用场景：
- 多路检索结果融合
- 文档重排序  
- 信息检索系统优化
"""

import logging
logger = logging.getLogger(__name__)

# 导入基于真实OpenMatch/FiT5的实现
try:
    from .fit5_fusion import (
        FiT5FusionEngine,
        FiT5Config,
        FusionResult,
        create_fit5_fusion_engine
    )
    FIT5_AVAILABLE = True
    logger.info("✅ FiT5融合引擎加载成功（基于OpenMatch/FiT5官方代码库）")
except ImportError as e:
    logger.error(f"FiT5融合引擎导入失败: {e}")
    FIT5_AVAILABLE = False
    FiT5FusionEngine = None
    FiT5Config = None
    FusionResult = None
    create_fit5_fusion_engine = None



__all__ = [
    "FiT5FusionEngine",
    "FiT5Config", 
    "FusionResult",
    "create_fit5_fusion_engine",
    "FIT5_AVAILABLE"
]
