"""
Nano GraphRAG - 增强版图检索增强生成系统 (整合版)

一个高性能、模块化的RAG系统，集成了：
- Modern Evaluator：基于RAGAS的现代评估体系 (整合传统+现代评估)
- Simple Optimizer：轻量级自适应优化器
- Complexity Router：智能复杂度感知路由
- Confidence Calibration：置信度校准系统

v0.2.0 整合版特性：
- 统一评估接口：整合传统与现代评估功能
- 统一融合接口：整合传统融合策略
- 保持向后兼容性
- 提供便捷的创建函数
"""

__version__ = "0.2.0"
__author__ = "Nano GraphRAG Team"

# 核心类 - 向后兼容
from .graphrag import (
    GraphRAG,                    # 向后兼容的主类
    EnhancedGraphRAG,           # 新的增强版主类
    create_enhanced_graphrag,    # 创建增强版实例
    create_basic_graphrag,       # 创建基础版实例
)

# 基础数据结构和接口
from .base import (
    QueryParam,
    BaseVectorStorage,
    BaseKVStorage,
    BaseGraphStorage,
    CommunitySchema,
    TextChunkSchema,
    StorageNameSpace,
)

# 存储实现
from ._storage import (
    JsonKVStorage,
    NetworkXStorage,
    NanoVectorDBStorage,
    # 新增存储实现
    FAISSVectorStorage,
    Neo4jStorage,
    HNSWVectorStorage,
    create_faiss_storage,
)

# 核心处理模块
from .chunking import get_chunks, chunking_by_token_size
from .entity_extraction import extract_entities
from .community import generate_community_report

# 查询处理
from .query_processing import (
    local_query,
    global_query, 
    naive_query,
)

# 复杂度分析和路由 (新增)
from .complexity import (
    ComplexityClassifier,
    ComplexityClassifierConfig,
    ComplexityAwareRouter,
)

# ============= 增强功能模块 (整合版) =============

# FiT5融合策略已移除，使用RRF置信度感知融合



# 简单的占位符函数
def create_evaluator(evaluator_type: str = "modern", **kwargs):
    """
    创建评估器实例
    
    Args:
        evaluator_type: 评估器类型，目前支持 "modern"
        **kwargs: 评估器配置参数
        
    Returns:
        评估器实例
    """
    if evaluator_type == "modern":
        # 导入现代评估器
        try:
            from .evaluation import ModernEvaluator, ModernEvaluatorConfig
            
            # 创建配置
            config = ModernEvaluatorConfig(**kwargs)
            evaluator = ModernEvaluator(config)
            
            logger.info("现代评估器创建成功")
            return evaluator
            
        except ImportError as e:
            logger.warning(f"现代评估器导入失败: {e}")
            return None
    else:
        logger.warning(f"不支持的评估器类型: {evaluator_type}")
        return None

# 检索结果对齐 (从实际实现的模块导入)
from .retrieval.alignment import (
    RetrievalResult,
    RetrievalAdapter,
)

# 工具函数
from ._utils import (
    logger,  # 添加logger导入
    compute_args_hash,
    compute_mdhash_id,
    normalize_text,
    compute_text_hash,
    get_timestamp,
    save_json,
    load_json,
)

# LLM和嵌入函数
from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_gpt_4o_mini_complete,
    azure_openai_embedding,
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    siliconflow_embedding,
)

# 配置管理
from .config import (
    load_config,
    save_config,
    create_config_from_template,
    validate_config,
    get_default_config,
)

# ============= 主要导出列表 =============
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 主要类
    "GraphRAG",
    "EnhancedGraphRAG", 
    "create_enhanced_graphrag",
    "create_basic_graphrag",
    "create_nano_graphrag",
    
    # 基础接口
    "QueryParam",
    "BaseVectorStorage",
    "BaseKVStorage", 
    "BaseGraphStorage",
    "CommunitySchema",
    "TextChunkSchema",
    "StorageNameSpace",
    
    # 存储实现
    "JsonKVStorage",
    "NetworkXStorage",
    "NanoVectorDBStorage",
    "FAISSVectorStorage",
    "Neo4jStorage", 
    "HNSWVectorStorage",
    "create_faiss_storage",
    
    # 核心处理
    "get_chunks",
    "chunking_by_token_size",
    "extract_entities",
    "generate_community_report",
    
    # 查询处理
    "local_query",
    "global_query",
    "naive_query",
    
    # 复杂度分析 (新增)
    "ComplexityClassifier",
    "ComplexityClassifierConfig", 
    "ComplexityAwareRouter",
    
    # ============= 增强功能 (整合版) =============

    # 评估系统 (现代评估器)
    "ModernEvaluator",
    "ModernEvaluatorConfig", 
    "create_evaluator",  # 占位符函数
    
    # 检索结果对齐 (已实现)
    "RetrievalResult",
    "RetrievalAdapter",
    
    # 混合检索功能已集成到其他模块中
    
    # 工具函数
    "compute_args_hash",
    "compute_mdhash_id",
    "normalize_text",
    "compute_text_hash",
    "get_timestamp",
    "save_json", 
    "load_json",
    
    # LLM函数
    "gpt_4o_complete",
    "gpt_4o_mini_complete",
    "openai_embedding",
    "siliconflow_embedding",
    
    # 配置管理
    "load_config",
    "save_config",
    "get_default_config",
]

# ============= 便捷创建函数 =============

def create_nano_graphrag(enhanced: bool = True, **kwargs):
    """
    便捷函数：创建nano graphrag实例
    
    Args:
        enhanced: 是否启用增强功能（默认True）
        **kwargs: 其他配置参数
    
    Returns:
        EnhancedGraphRAG实例
    
    Examples:
        >>> # 创建增强版（推荐）
        >>> rag = create_nano_graphrag()
        
        >>> # 创建基础版（兼容老代码）
        >>> rag = create_nano_graphrag(enhanced=False)
        
        >>> # 自定义配置
        >>> rag = create_nano_graphrag(
        ...     working_dir="./my_graphrag",
        ...     enable_naive_rag=True
        ... )
    """
    if enhanced:
        return create_enhanced_graphrag(**kwargs)
    else:
        return create_basic_graphrag(**kwargs)

def create_unified_pipeline(evaluator_type: str = "modern", **kwargs):
    """
    创建统一的RRF融合+评估管道
    
    Args:
        evaluator_type: 评估器类型 ("modern", "comprehensive", "basic")
        **kwargs: 配置参数
        
    Returns:
        (fusion_engine, evaluator) 元组
    """
    from .retrieval import create_fusion_engine
    
    # 创建RRF融合引擎
    fusion_engine = create_fusion_engine(**kwargs)
    
    # 创建评估器
    evaluator = create_evaluator(evaluator_type, **kwargs)
    
    return fusion_engine, evaluator

def get_available_fusion_types():
    """获取可用的融合类型"""
    available_types = []
    
    # 检查RRF融合是否可用
    try:
        from .retrieval import ConfidenceAwareFusion
        available_types.append("rrf")
    except ImportError:
        pass
    
    # 默认总是有简单的线性融合作为回退
    if "rrf" not in available_types:
        available_types.append("fallback")
    
    return available_types

# 系统能力标志
RRF_FUSION_AVAILABLE = True  # RRF融合总是可用
BASIC_EVALUATION_AVAILABLE = True  # 基础评估可用

def get_system_capabilities():
    """获取系统可用能力"""
    capabilities = {
        "fusion_types": get_available_fusion_types(),
        "rrf_fusion_available": RRF_FUSION_AVAILABLE,
        "basic_evaluation_available": BASIC_EVALUATION_AVAILABLE,
        "enhanced_features": [
            "modern_evaluator", 
            "complexity_router",
            "rrf_fusion",
            "retrieval_alignment"
        ]
    }
    return capabilities

# 添加便捷函数到导出列表
__all__.extend([
    "create_unified_pipeline", 
    "get_system_capabilities"
])

# ============= 依赖检查和初始化 =============

def check_dependencies():
    """检查依赖项是否正确安装"""
    import warnings
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        warnings.warn("PyTorch not found. Enhanced features may not work.")
    
    try:
        import sentence_transformers
        logger.info(f"Sentence Transformers version: {sentence_transformers.__version__}")
    except ImportError:
        warnings.warn("Sentence Transformers not found. Modern evaluator may not work.")
    
    try:
        import sklearn
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        warnings.warn("Scikit-learn not found. Some evaluation features may not work.")

# 初始化时检查依赖
try:
    check_dependencies()
except Exception as e:
    logger.warning(f"Dependency check failed: {e}")

# 显示欢迎信息
logger.info(f"Nano GraphRAG v{__version__} (置信度感知融合版) loaded successfully!")
logger.info("🔧 核心特性: RRF置信度感知融合, Modern Evaluator")
logger.info("🚀 融合技术: 基于互惠排名融合的智能多源检索")
logger.info("💡 快速开始: create_nano_graphrag() | 管道创建: create_unified_pipeline()")
logger.info(f"📊 系统能力: {len(get_system_capabilities()['fusion_types'])} 种融合策略可用")
