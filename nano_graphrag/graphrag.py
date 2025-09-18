"""
Enhanced GraphRAG System
集成现代评估器、复杂度感知路由与传统融合策略，适配本科生项目的实际需求，保持高性能和可用性
"""

import os
import asyncio
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Type, Union, List, Dict, Any, Optional, Callable

import tiktoken

from ._utils import (
    logger,
    always_get_an_event_loop,
    limit_async_func_call,
    convert_response_to_json,
    compute_mdhash_id,
)

# 核心存储和处理模块
from .base import (
    BaseVectorStorage,
    BaseKVStorage, 
    BaseGraphStorage,
    QueryParam,
    CommunitySchema,
    TextChunkSchema,
    StorageNameSpace,
)

# 数据处理模块
from .chunking import get_chunks, chunking_by_token_size
from .entity_extraction import extract_entities
from .community import generate_community_report

# 查询处理模块
from .query_processing import local_query, global_query, naive_query

# 存储实现
from ._storage import JsonKVStorage, NetworkXStorage, NanoVectorDBStorage
from ._storage.other.bm25 import BM25Storage  # 新增：BM25 存储后端

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
    qwen_turbo_complete,
    siliconflow_embedding,
)

from .complexity.router import ComplexityAwareRouter
from .retrieval import ConfidenceAwareFusion, FusionConfig, create_fusion_engine


@dataclass
class EnhancedGraphRAG:
    
    # 基础配置
    working_dir: str = field(
        # Windows 路径不允许冒号，统一使用无冒号的时间格式
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 模式控制
    enable_local: bool = True
    enable_naive_rag: bool = True  # 默认启用，增加灵活性
    enable_bm25: bool = False  # 新增：是否启用 BM25 检索
    enable_enhanced_features: bool = True  # 是否启用增强功能
    
    # 文本分块配置
    chunk_func: Callable = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    
    # 实体提取配置
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    
    # 图聚类配置
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    
    # 节点嵌入配置（保留兼容性）
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    
    # 社区报告配置
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    
    # 嵌入配置（默认使用硅基流动 BGE-M3）
    embedding_func: Callable = field(default_factory=lambda: siliconflow_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
    
    # LLM配置
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16
    
    # 实体提取函数
    entity_extraction_func: callable = extract_entities
    
    # DSPy实体提取配置
    use_compiled_dspy_entity_relationship: bool = False  # 禁用编译模型避免字段冲突
    
    # 存储配置
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    # 运行时存储实例（新增：BM25）
    bm25_storage: Optional[BM25Storage] = None
    
    # 增强功能配置 - 现代评估器
    enable_modern_evaluator: bool = True  # 启用现代评估器
    evaluator_config: Optional[Dict[str, Any]] = None  # 评估器配置

    # RRF置信度感知融合配置
    fusion_config: Optional[FusionConfig] = None  # RRF融合配置
    enable_confidence_fusion: bool = True  # 是否启用置信度感知融合
    rrf_k: float = 60.0  # RRF平滑参数
    fusion_max_results: int = 20  # 融合最大结果数
    
    # 渐进式检索配置
    confidence_high_threshold: float = 0.9  # 高置信度阈值（策略A）
    confidence_medium_threshold: float = 0.6  # 中等置信度阈值（策略B）
    max_parallel_retrievers: int = 4  # 最大并行检索器数量
    retrieval_timeout_seconds: float = 30.0  # 检索超时时间（秒）
    
    # 检索器优先级配置
    retriever_priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "global": 1.0,    # 全局图检索权重
        "local": 0.9,     # 局部图检索权重
        "naive": 0.8,     # 向量检索权重
        "bm25": 0.7,      # BM25检索权重
        "llm_only": 0.6   # 纯LLM权重
    })
    
    # 扩展配置
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # 混合检索/路由（与 Hybrid 扩展兼容的占位属性）
    router_cls: type = ComplexityAwareRouter  # 供扩展类检查
    router_kwargs: dict = field(default_factory=dict)  # 供扩展类复制并覆盖
    model_path: str = field(default_factory=lambda: "nano_graphrag/models/modernbert_complexity_classifier")

    def __post_init__(self):

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"Enhanced GraphRAG init with param:\n\n  {_print_config}\n")

        # 配置LLM提供商
        self._setup_llm_providers()
        
        # 创建工作目录
        self._setup_working_directory()
        
        # 初始化存储组件
        self._setup_storage_components()
        
        # 初始化LLM和嵌入函数
        self._setup_llm_and_embedding()
        
        # 初始化增强功能模块
        if self.enable_enhanced_features:
            self._setup_enhanced_modules()
    
    def _setup_llm_providers(self):
        """设置LLM提供商"""
        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info("Switched to Azure OpenAI")

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info("Switched to Amazon Bedrock")

        # DashScope/OpenAI 兼容端点处理
        # 1) 检测 DashScope 端点时，移除会导致 JSON mode 的参数
        # 2) 若检测到 DASHSCOPE_API_KEY，则将社区报告的默认模型函数切到 qwen-turbo
        api_base = os.getenv("OPENAI_BASE_URL", "") or os.getenv("OPENAI_API_BASE", "")
        if "dashscope" in api_base.lower() or os.getenv("DASHSCOPE_API_KEY", ""):
            if isinstance(self.special_community_report_llm_kwargs, dict):
                self.special_community_report_llm_kwargs.pop("response_format", None)
            logger.info("Detected DashScope endpoint, removed response_format from community report kwargs.")
            # 切换社区报告到 Qwen Turbo（若仍是默认的 gpt_4o_complete）
            try:
                if self.best_model_func == gpt_4o_complete:
                    self.best_model_func = qwen_turbo_complete
                    logger.info("Switch community report LLM to qwen-turbo for DashScope.")
            except Exception:
                pass
    
    def _setup_working_directory(self):
        """设置工作目录"""
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
    
    def _setup_storage_components(self):
        """设置存储组件"""
        config_dict = asdict(self)
        
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=config_dict
        )
        
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=config_dict
        )
        
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=config_dict
            )
            if self.enable_llm_cache
            else None
        )
        
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=config_dict
        )
        
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=config_dict
        )

        # 新增：初始化 BM25 存储（可选）
        if self.enable_bm25:
            try:
                self.bm25_storage = BM25Storage(namespace="bm25", global_config=config_dict)
                # 兼容旧代码命名（例如 examples 中使用 bm25_store）
                setattr(self, "bm25_store", self.bm25_storage)
                logger.info("BM25 storage initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25 storage: {e}")
                self.bm25_storage = None
                setattr(self, "bm25_store", None)
        else:
            self.bm25_storage = None
            setattr(self, "bm25_store", None)
    
    def _setup_llm_and_embedding(self):
        """设置LLM和嵌入函数"""
        # 限制并发调用
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        # 设置向量数据库
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        # 设置LLM函数
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            self.best_model_func
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            self.cheap_model_func
        )
    
    def _setup_enhanced_modules(self):
        """设置增强功能模块"""
        try:
            # 设置默认融合策略
            fusion_strategy = "fit5" if self.enable_fit5_fusion else "traditional"
            logger.info(f"Using fusion strategy: {fusion_strategy}")
            
            # Modern Evaluator 初始化
            if self.enable_modern_evaluator:
                try:
                    from .evaluation import ModernEvaluator, ModernEvaluatorConfig, EVALUATOR_AVAILABLE
                    
                    if EVALUATOR_AVAILABLE:
                        # 创建评估器配置
                        if self.evaluator_config:
                            evaluator_config = ModernEvaluatorConfig(**self.evaluator_config)
                        else:
                            evaluator_config = ModernEvaluatorConfig()
                        
                        self.modern_evaluator = ModernEvaluator(evaluator_config)
                        logger.info("Modern Evaluator初始化成功")
                    else:
                        logger.warning("评估器依赖库不可用，Modern Evaluator禁用")
                        self.modern_evaluator = None
                        
                except Exception as e:
                    logger.warning(f"Modern Evaluator初始化失败: {e}")
                    self.modern_evaluator = None
            else:
                self.modern_evaluator = None
                logger.info("Modern Evaluator已禁用")
            
            # Complexity Router
            self.complexity_router = ComplexityAwareRouter()
            # 兼容：暴露通用名称供 Hybrid 扩展使用
            self.router = self.complexity_router
            logger.info("Complexity Router initialized successfully")
            
            # RRF置信度感知融合引擎初始化
            if self.enable_confidence_fusion:
                if self.fusion_config is None:
                    self.fusion_config = FusionConfig(
                        k=self.rrf_k,
                        max_results=self.fusion_max_results,
                        confidence_aware=True
                    )
                
                self.fusion_engine = create_fusion_engine(
                    k=self.fusion_config.k,
                    max_results=self.fusion_config.max_results,
                    confidence_aware=self.fusion_config.confidence_aware
                )
                logger.info("✅ RRF置信度感知融合引擎初始化成功")
                logger.info(f"RRF参数k: {self.fusion_config.k}")
                logger.info(f"最大结果数: {self.fusion_config.max_results}")
            else:
                self.fusion_engine = None
                logger.info("RRF融合引擎已禁用")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced modules: {e}")
            logger.warning("Falling back to basic functionality")
            self.enable_enhanced_features = False

    def insert(self, string_or_strings):
        """同步插入接口"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """同步查询接口"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询接口 - 渐进式并行检索架构
        
        实现三个阶段的处理流程：
        1. 复杂度和置信度分析
        2. 渐进式检索策略执行
        3. 智能融合
        """
        try:
            # 第一阶段：复杂度和置信度分析
            if self.enable_enhanced_features and hasattr(self, 'complexity_router'):
                complexity_result = await self.complexity_router.predict_complexity_detailed(query)
                logger.info(f"复杂度分析: {complexity_result.get('complexity')}, 置信度: {complexity_result.get('confidence', 0):.3f}")
            else:
                # 回退到默认复杂度结果
                complexity_result = {
                    "complexity": "one_hop",
                    "confidence": 0.5,
                    "probabilities": {"one_hop": 1.0},
                    "method": "fallback"
                }
                logger.warning("复杂度分析器不可用，使用默认配置")
            
            # 第二阶段：渐进式检索策略执行
            retrieval_tasks = self._plan_retrieval_tasks(complexity_result, param)
            
            if not retrieval_tasks:
                # 策略A：直接LLM回答（当检索任务为空时）
                logger.info("执行策略A - 直接LLM回答")
                response = await self._llm_only_response(query, param)
                await self._query_done()
                return response
            
            # 并行执行检索任务
            retrieval_results = await self._execute_retrieval_tasks(retrieval_tasks, query, param)
            
            # 第三阶段：智能融合
            if len(retrieval_results) == 1:
                # 单一检索结果，直接处理
                mode, result = next(iter(retrieval_results.items()))
                if mode == "llm_only":
                    # llm_only返回字符串
                    response = result
                else:
                    # 其他模式需要转换为字符串响应
                    response = await self._convert_retrieval_results_to_response(result, query, param)
                logger.info(f"单一检索模式完成: {mode}")
            else:
                # 多检索结果，需要融合
                response = await self._confidence_aware_fusion(retrieval_results, complexity_result, query, param)
                logger.info(f"多检索模式融合完成: {list(retrieval_results.keys())}")
            
            await self._query_done()
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            # 回退到传统单一模式
            logger.info("回退到传统单一模式")
            response = await self._fallback_single_mode_query(query, param)
            await self._query_done()
            return response
    
    async def _convert_retrieval_results_to_response(self, retrieval_results, query: str, param: QueryParam) -> str:
        """
        将检索结果转换为最终响应
        
        Args:
            retrieval_results: RetrievalResult列表或上下文字符串
            query: 查询文本
            param: 查询参数
            
        Returns:
            最终的响应字符串
        """
        try:
            from .retrieval.alignment import RetrievalResult
            
            if isinstance(retrieval_results, str):
                # 已经是字符串格式的上下文，直接返回
                return retrieval_results
            elif isinstance(retrieval_results, list) and retrieval_results:
                # RetrievalResult列表，需要转换
                if isinstance(retrieval_results[0], RetrievalResult):
                    # 提取内容并生成回答
                    context_parts = [result.content for result in retrieval_results if result.content]
                    if not context_parts:
                        from .answer_generation.prompts import PROMPTS
                        return PROMPTS["fail_response"]
                    
                    context = "\n\n".join(context_parts)
                    
                    # 使用LLM生成最终回答
                    from .answer_generation.prompts import PROMPTS
                    sys_prompt_temp = PROMPTS.get("naive_rag_response", "Please answer based on the context: {content_data}")
                    sys_prompt = sys_prompt_temp.format(
                        content_data=context, 
                        response_type=param.response_type
                    )
                    
                    response = await self.best_model_func(
                        query,
                        system_prompt=sys_prompt,
                    )
                    return response
                else:
                    # 未知格式，转为字符串
                    return str(retrieval_results[0]) if retrieval_results else ""
            else:
                # 空结果
                from .answer_generation.prompts import PROMPTS
                return PROMPTS["fail_response"]
                
        except Exception as e:
            logger.error(f"结果转换失败: {e}")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]
    
    async def _fallback_single_mode_query(self, query: str, param: QueryParam) -> str:
        """
        回退到传统单一模式查询
        
        Args:
            query: 查询文本
            param: 查询参数
            
        Returns:
            查询响应
        """
        try:
            # 验证模式可用性
            if param.mode == "local" and not self.enable_local:
                param.mode = "naive"
            if param.mode == "naive" and not self.enable_naive_rag:
                param.mode = "llm_only"
            
            # 执行传统查询
            if param.mode == "local":
                return await local_query(
                    query, self.chunk_entity_relation_graph, self.entities_vdb,
                    self.community_reports, self.text_chunks, param, self._get_query_config()
                )
            elif param.mode == "global":
                return await global_query(
                    query, self.chunk_entity_relation_graph, self.entities_vdb,
                    self.community_reports, self.text_chunks, param, self._get_query_config()
                )
            elif param.mode == "naive":
                return await naive_query(
                    query, self.chunks_vdb, self.text_chunks, param, self._get_query_config()
                )
            elif param.mode == "bm25":
                from .query_processing.bm25_query import bm25_query
                return await bm25_query(
                    query, self.bm25_storage, self.text_chunks, param, self._get_query_config()
                )
            elif param.mode == "llm_only":
                return await self._llm_only_response(query, param)
            else:
                return await self._llm_only_response(query, param)
                
        except Exception as e:
            logger.error(f"回退查询失败: {e}")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]

    async def ainsert(self, string_or_strings):
        """异步插入接口"""
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                
            # 处理新文档
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # 文本分块
            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # 新增：BM25 文档索引（基于 chunks 内容）
            if self.enable_bm25 and self.bm25_storage is not None:
                try:
                    bm25_docs = {k: v.get("content", "") for k, v in inserting_chunks.items()}
                    # 过滤空内容，避免无效索引
                    bm25_docs = {k: c for k, c in bm25_docs.items() if isinstance(c, str) and c.strip()}
                    if bm25_docs:
                        logger.info("Indexing %d chunks into BM25 storage", len(bm25_docs))
                        await self.bm25_storage.index_documents(bm25_docs)
                except Exception as e:
                    logger.warning(f"BM25 indexing failed: {e}")

            # 清理社区报告（需要重新生成）
            await self.community_reports.drop()

            # 实体提取和图构建
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            # 图聚类和社区报告生成
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # 提交更新
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            
        finally:
            await self._insert_done()

    async def evaluate_system(self, 
                            questions: List[str],
                            answers: List[str] = None,
                            contexts_list: List[List[str]] = None,
                            ground_truths: List[str] = None,
                            system_name: str = "GraphRAG System") -> Dict[str, Any]:
        """
        评估系统性能
        使用现代评估器进行全面评估
        
        Args:
            questions: 问题列表
            answers: 答案列表（可选，如果为空则会调用系统生成）
            contexts_list: 上下文列表（可选）
            ground_truths: 标准答案列表（必需）
            system_name: 系统名称
        
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.enable_enhanced_features or not hasattr(self, 'modern_evaluator') or self.modern_evaluator is None:
            logger.warning("Modern evaluator not available, skipping evaluation")
            return {"error": "Modern evaluator not available"}
        
        try:
            # 如果没有提供标准答案，无法进行评估
            if not ground_truths or len(ground_truths) != len(questions):
                logger.error("Ground truth answers are required for evaluation")
                return {"error": "Ground truth answers are required"}
            
            # 构建测试问题
            test_questions = []
            for i, question in enumerate(questions):
                test_item = {
                    "question": question,
                    "ground_truth": ground_truths[i]
                }
                test_questions.append(test_item)
            
            # 使用现代评估器进行评估
            evaluation_result = await self.modern_evaluator.evaluate_system(
                rag_system=self,
                test_questions=test_questions,
                system_name=system_name
            )
            
            # 转换为字典格式返回
            result_dict = {
                "system_name": evaluation_result.system_name,
                "total_cases": evaluation_result.total_cases,
                "valid_cases": evaluation_result.valid_cases,
                "average_metrics": evaluation_result.average_metrics,
                "execution_time": evaluation_result.execution_time,
                "evaluation_config": evaluation_result.evaluation_config,
                "individual_results": [
                    {
                        "case_id": r.case_id,
                        "metrics": r.metrics,
                        "metadata": r.metadata
                    } for r in evaluation_result.individual_results
                ]
            }
            
            logger.info("System evaluation completed successfully")
            return result_dict
            
        except Exception as e:
            logger.error(f"System evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "basic_info": {
                "working_dir": self.working_dir,
                "enable_enhanced_features": self.enable_enhanced_features,
                "enable_local": self.enable_local,
                "enable_naive_rag": self.enable_naive_rag
            }
        }
        
        # 添加增强功能统计
        if self.enable_enhanced_features:
            if hasattr(self, 'complexity_router') and self.complexity_router:
                stats["complexity_stats"] = self.complexity_router.get_complexity_stats()
            
            if hasattr(self, 'modern_evaluator') and self.modern_evaluator:
                stats["evaluation_summary"] = self.modern_evaluator.get_evaluation_summary()
            
            # 添加融合引擎统计
            if hasattr(self, 'fusion_engine') and self.fusion_engine:
                stats["fusion_stats"] = self.fusion_engine.get_fusion_stats()
                stats["fusion_engine_type"] = "RRF_ConfidenceAware"
        
        return stats

    async def _insert_start(self):
        """插入开始回调"""
        tasks = []
        for storage_inst in [self.chunk_entity_relation_graph]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        """插入完成回调"""
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.bm25_storage,  # 新增：BM25 索引落盘
        ]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_done_callback())
        await asyncio.gather(*tasks)

    def _plan_retrieval_tasks(self, complexity_result: Dict[str, Any], param: QueryParam) -> List[str]:
        """
        规划检索任务 - 基于复杂度和置信度选择检索策略
        
        Args:
            complexity_result: 复杂度分析结果
            param: 查询参数
            
        Returns:
            选择的检索模式列表
        """
        # 获取可用的检索模式
        available_modes = ["llm_only", "naive", "bm25", "local", "global"]
        if not self.enable_local:
            available_modes = [m for m in available_modes if m not in ["local", "global"]]
        if not self.enable_naive_rag:
            available_modes = [m for m in available_modes if m != "naive"]
        if not self.enable_bm25:
            available_modes = [m for m in available_modes if m != "bm25"]
        
        # 使用ComplexityRouter规划检索任务
        if hasattr(self, 'complexity_router') and self.complexity_router:
            try:
                retrieval_plan = self.complexity_router.get_retrieval_plan(complexity_result, available_modes)
                logger.info(f"检索计划: {retrieval_plan}")
                return retrieval_plan
            except Exception as e:
                logger.error(f"检索规划失败: {e}")
                # 回退到单一模式
                return [param.mode] if param.mode in available_modes else [available_modes[0]]
        else:
            # 如果没有路由器，回退到原始模式
            return [param.mode] if param.mode in available_modes else [available_modes[0]]
    
    async def _execute_retrieval_tasks(self, retrieval_modes: List[str], query: str, param: QueryParam) -> Dict[str, Any]:
        """
        并行执行检索任务
        
        Args:
            retrieval_modes: 要执行的检索模式列表
            query: 查询文本
            param: 查询参数
            
        Returns:
            检索结果字典 {mode: result}
        """
        import asyncio
        from .query_processing.bm25_query import bm25_query
        from .query_processing.llm_only_query import llm_only_query
        
        async def _execute_single_retrieval(mode: str):
            """执行单个检索任务"""
            try:
                if mode == "local":
                    result = await local_query(
                        query,
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.community_reports,
                        self.text_chunks,
                        param,
                        self._get_query_config(),
                        return_raw_results=True  # 使用新的参数
                    )
                elif mode == "global":
                    result = await global_query(
                        query,
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.community_reports,
                        self.text_chunks,
                        param,
                        self._get_query_config(),
                        return_raw_results=True
                    )
                elif mode == "naive":
                    result = await naive_query(
                        query,
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        self._get_query_config(),
                        return_raw_results=True
                    )
                elif mode == "bm25":
                    result = await bm25_query(
                        query,
                        self.bm25_storage if hasattr(self, 'bm25_storage') else None,
                        self.text_chunks,
                        param,
                        self._get_query_config(),
                        return_raw_results=True
                    )
                elif mode == "llm_only":
                    # LLM only模式返回字符串而不是RetrievalResult列表
                    result = await llm_only_query(
                        query,
                        param,
                        self._get_query_config(),
                    )
                else:
                    logger.warning(f"Unknown retrieval mode: {mode}")
                    result = []
                
                return result
                
            except Exception as e:
                logger.error(f"检索模式 {mode} 执行失败: {e}")
                return []
        
        # 并行执行所有检索任务
        tasks = {mode: _execute_single_retrieval(mode) for mode in retrieval_modes}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # 构建结果字典
        retrieval_results = {}
        for i, mode in enumerate(retrieval_modes):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"检索模式 {mode} 出现异常: {result}")
                retrieval_results[mode] = []
            else:
                retrieval_results[mode] = result
        
        logger.info(f"并行检索完成，模式: {list(retrieval_results.keys())}")
        return retrieval_results
    
    async def _llm_only_response(self, query: str, param: QueryParam) -> str:
        """
        处理策略A的直接LLM回答
        
        Args:
            query: 查询文本
            param: 查询参数
            
        Returns:
            LLM生成的回答
        """
        try:
            from .query_processing.llm_only_query import llm_only_query
            response = await llm_only_query(
                query,
                param,
                self._get_query_config(),
            )
            logger.info("使用LLM直接回答策略")
            return response
        except Exception as e:
            logger.error(f"LLM直接回答失败: {e}")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]
    
    async def _confidence_aware_fusion(self, retrieval_results: Dict[str, Any], complexity_result: Dict[str, Any], query: str, param: QueryParam) -> str:
        """
        置信度感知的多源检索融合方法
        
        使用RRF(Reciprocal Rank Fusion)算法和置信度感知权重调整，
        将多个检索器的结果进行智能重排序和融合。
        
        核心特性：
        - RRF排序融合：基于排名而非分数进行融合，避免不同检索器分数不可比的问题
        - 置信度感知：根据查询复杂度动态调整各检索源的权重
        - 多样性保证：确保融合结果来源多样化
        
        Args:
            retrieval_results: 检索结果字典 {retriever_name: List[RetrievalResult] or str}
            complexity_result: 复杂度分析结果
            query: 查询文本
            param: 查询参数
            
        Returns:
            融合后的响应字符串
        """
        try:
            logger.info("使用RRF置信度感知融合引擎")
            
            # 检查是否启用置信度感知融合引擎
            if not self.enable_confidence_fusion or self.fusion_engine is None:
                logger.warning("RRF融合引擎未启用，回退到简单融合")
                return await self._fallback_fusion_strategy(retrieval_results, query, param)
            
            # 将检索结果转换为按源分组的RetrievalResult列表
            from .retrieval.alignment import RetrievalResult
            results_by_source = {}
            
            for retriever_name, result in retrieval_results.items():
                if not result:
                    continue
                    
                if isinstance(result, str):
                    # 字符串结果转换为RetrievalResult列表
                    if retriever_name == "llm_only":
                        # llm_only结果直接返回，不参与融合
                        if len(retrieval_results) == 1:
                            return result
                        continue
                    else:
                        # 其他字符串结果转换为RetrievalResult
                        # 尝试按段落分割字符串结果以增加粒度
                        content_parts = []
                        if "--New Chunk--" in result:
                            content_parts = [s.strip() for s in result.split("--New Chunk--") if s.strip()]
                        elif "-----" in result and "```csv" in result:
                            # 处理图检索结果格式
                            parts = result.split("-----")
                            for part in parts:
                                if "```csv" in part:
                                    csv_start = part.find("```csv\n") + 7
                                    csv_end = part.find("\n```")
                                    if csv_start != -1 and csv_end != -1:
                                        csv_content = part[csv_start:csv_end].strip()
                                        if csv_content:
                                            content_parts.append(csv_content)
                        else:
                            content_parts = [result.strip()]
                        
                        retrieval_results_list = []
                        for i, content in enumerate(content_parts):
                            if content and len(content.strip()) > 10:
                                converted_result = RetrievalResult(
                                    content=content,
                                    score=1.0 - (i * 0.1),  # 递减分数
                                    source=retriever_name,
                                    chunk_id=f"{retriever_name}_part_{i}",
                                    rank=i + 1,
                                    metadata={"converted_from_string": True}
                                )
                                retrieval_results_list.append(converted_result)
                        results_by_source[retriever_name] = retrieval_results_list
                
                elif isinstance(result, list) and all(isinstance(r, RetrievalResult) for r in result):
                    # 已经是RetrievalResult列表
                    results_by_source[retriever_name] = result
                
                else:
                    logger.warning(f"未知的结果格式来自 {retriever_name}: {type(result)}")
                    continue
            
            if not results_by_source:
                logger.warning("没有有效的检索结果可供融合")
                from .answer_generation.prompts import PROMPTS
                return PROMPTS["fail_response"]
            
            # 使用RRF融合引擎进行智能融合
            fused_results = self.fusion_engine.fuse_results(
                results_by_source=results_by_source,
                query_complexity=complexity_result
            )
            
            if not fused_results:
                logger.warning("RRF融合引擎返回空结果")
                from .answer_generation.prompts import PROMPTS
                return PROMPTS["fail_response"]
            
            # 将融合结果转换为上下文并生成答案
            response = await self._generate_answer_from_rrf_results(fused_results, query, param, complexity_result)
            
            logger.info(f"RRF融合完成，使用了 {len(results_by_source)} 个检索器，"
                       f"融合得到 {len(fused_results)} 个结果")
            
            return response
            
        except Exception as e:
            logger.error(f"RRF融合失败: {e}")
            logger.info("回退到简单融合策略")
            return await self._fallback_fusion_strategy(retrieval_results, query, param)
    
    async def _generate_answer_from_rrf_results(
        self, 
        fused_results: List,  # List[RetrievalResult] from RRF
        query: str, 
        param: QueryParam,
        complexity_result: Dict[str, Any]
    ) -> str:
        """
        从RRF融合结果生成最终答案
        
        Args:
            fused_results: RRF融合后的检索结果列表
            query: 原始查询
            param: 查询参数
            complexity_result: 复杂度分析结果
            
        Returns:
            生成的答案字符串
        """
        try:
            from .retrieval.alignment import RetrievalResult
            
            if not fused_results:
                from .answer_generation.prompts import PROMPTS
                return PROMPTS["fail_response"]
            
            # 构建融合上下文
            context_parts = []
            for i, result in enumerate(fused_results):
                if isinstance(result, RetrievalResult):
                    # 添加结果来源和RRF分数信息
                    rrf_score = result.metadata.get('rrf_score', 0.0)
                    fusion_rank = result.metadata.get('fusion_rank', i + 1)
                    source_info = f"[来源: {result.source}, RRF分数: {rrf_score:.3f}, 融合排名: {fusion_rank}]"
                    context_parts.append(f"{source_info}\n{result.content}")
                else:
                    # 兼容其他格式
                    context_parts.append(str(result))
            
            if not context_parts:
                from .answer_generation.prompts import PROMPTS
                return PROMPTS["fail_response"]
            
            # 使用分隔符连接内容
            fused_context = "\n\n--融合内容--\n".join(context_parts)
            
            # 添加RRF融合统计信息
            fusion_stats = f"\n\n[RRF融合统计] 使用置信度感知的RRF算法融合了 {len(fused_results)} 个结果"
            if self.fusion_engine:
                fusion_engine_stats = self.fusion_engine.get_fusion_stats()
                fusion_stats += f"，融合引擎已处理 {fusion_engine_stats.get('total_fusions', 0)} 次融合"
            
            complete_context = fused_context + fusion_stats
            
            # 根据复杂度选择合适的提示模板
            confidence = complexity_result.get("confidence", 0.5)
            complexity = complexity_result.get("complexity", "one_hop")
            
            # 选择提示模板
            from .answer_generation.prompts import PROMPTS
            if complexity == "multi_hop" or confidence < 0.5:
                # 复杂查询或低置信度，使用更详细的提示
                prompt_template = PROMPTS.get("fusion_complex_response", PROMPTS["naive_rag_response"])
            else:
                # 简单查询或高置信度，使用标准提示
                prompt_template = PROMPTS.get("fusion_response", PROMPTS["naive_rag_response"])
            
            # 格式化提示
            system_prompt = prompt_template.format(
                content_data=complete_context,
                response_type=param.response_type
            )
            
            # 生成答案
            use_model_func = self.best_model_func
            response = await use_model_func(
                query,
                system_prompt=system_prompt,
                **self.special_community_report_llm_kwargs
            )
            
            logger.debug(f"从 {len(fused_results)} 个RRF融合结果生成答案成功")
            return response
            
        except Exception as e:
            logger.error(f"从RRF融合结果生成答案失败: {e}")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]
    
    
    async def _fallback_fusion_strategy(
        self, 
        retrieval_results: Dict[str, Any], 
        query: str, 
        param: QueryParam
    ) -> str:
        """
        回退融合策略 - 当置信度感知融合失败时使用
        
        Args:
            retrieval_results: 检索结果字典
            query: 查询文本
            param: 查询参数
            
        Returns:
            回退策略的响应
        """
        try:
            logger.info("使用回退融合策略")
            
            # 简单策略：按优先级选择第一个非空结果
            priority_order = ["global", "local", "naive", "bm25", "llm_only"]
            
            for mode in priority_order:
                if mode in retrieval_results and retrieval_results[mode]:
                    result = retrieval_results[mode]
                    logger.info(f"回退到 {mode} 检索结果")
                    
                    if mode == "llm_only":
                        return result
                    else:
                        return await self._convert_retrieval_results_to_response(result, query, param)
            
            # 如果按优先级没找到，选择任何非空结果
            for mode, result in retrieval_results.items():
                if result:
                    logger.info(f"回退到 {mode} 检索结果（无优先级匹配）")
                    if mode == "llm_only":
                        return result
                    else:
                        return await self._convert_retrieval_results_to_response(result, query, param)
            
            # 如果所有结果都为空，返回失败响应
            logger.warning("所有检索结果都为空")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]
            
        except Exception as e:
            logger.error(f"回退融合策略失败: {e}")
            from .answer_generation.prompts import PROMPTS
            return PROMPTS["fail_response"]

    def _get_query_config(self) -> Dict[str, Any]:
        """获取查询配置字典，避免asdict的序列化问题"""
        return {
            "best_model_func": self.best_model_func,
            "cheap_model_func": self.cheap_model_func,
            "convert_response_to_json_func": self.convert_response_to_json_func,
            "embedding_func": self.embedding_func,
            "entity_extraction_func": self.entity_extraction_func,
            "best_model_max_token_size": self.best_model_max_token_size,
            "cheap_model_max_token_size": self.cheap_model_max_token_size,
            "tiktoken_model_name": self.tiktoken_model_name,
            "special_community_report_llm_kwargs": self.special_community_report_llm_kwargs,
            "llm_response_cache": self.llm_response_cache, # 新增
            # 添加其他查询函数可能需要的配置
            "embedding_batch_num": self.embedding_batch_num,
            "embedding_func_max_async": self.embedding_func_max_async,
            "query_better_than_threshold": self.query_better_than_threshold,
        }

    async def _query_done(self):
        """查询完成回调"""
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_done_callback())
        await asyncio.gather(*tasks)


# 为了向后兼容，保留原有的GraphRAG类
GraphRAG = EnhancedGraphRAG


def create_enhanced_graphrag(**kwargs) -> EnhancedGraphRAG:

    return EnhancedGraphRAG(**kwargs)


def create_basic_graphrag(**kwargs) -> EnhancedGraphRAG:

    kwargs['enable_enhanced_features'] = False
    return EnhancedGraphRAG(**kwargs)
