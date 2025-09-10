"""
现代RAG评估器

基于RAGAS框架的标准评估实现，与GraphRAG系统无缝集成。
提供学术界和工业界公认的评估指标。
"""

import asyncio
import os
import time
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict

# 核心依赖检查
RAGAS_AVAILABLE = False
LANGCHAIN_AVAILABLE = False

try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy, 
        ContextPrecision,
        ContextRecall,
        SemanticSimilarity,
        AnswerCorrectness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"RAGAS库导入失败: {e}")

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain库导入失败: {e}")

import pandas as pd
from ..base import QueryParam

logger = logging.getLogger(__name__)

@dataclass
class EvaluationCase:
    """评估用例数据结构"""
    question: str
    ground_truth: str
    answer: str
    contexts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class EvaluationResult:
    """评估结果数据结构"""
    case_id: str
    metrics: Dict[str, float]
    raw_scores: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    system_name: str
    total_cases: int
    valid_cases: int
    average_metrics: Dict[str, float]
    individual_results: List[EvaluationResult]
    execution_time: float
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModernEvaluatorConfig:
    """现代评估器配置"""
    
    # API配置
    dashscope_api_key: Optional[str] = None
    siliconflow_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # 模型配置
    llm_model: str = "qwen-turbo"
    embedding_model: str = "BAAI/bge-m3"
    
    # 评估指标配置
    enable_faithfulness: bool = True
    enable_answer_relevancy: bool = True
    enable_context_precision: bool = True
    enable_context_recall: bool = True
    enable_semantic_similarity: bool = True
    enable_answer_correctness: bool = True
    
    # 性能配置
    batch_size: int = 5
    max_retries: int = 3
    timeout_seconds: float = 300.0
    
    # 输出配置
    save_detailed_results: bool = True
    save_individual_cases: bool = False
    output_format: str = "json"  # json, csv, both
    
    # 调试配置
    verbose: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """配置后处理"""
        # 从环境变量获取API密钥
        if not self.dashscope_api_key:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.siliconflow_api_key:
            self.siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

class ModernEvaluator:
    """
    现代RAG评估器
    
    基于RAGAS框架，提供标准化的RAG系统评估功能。
    与GraphRAG系统无缝集成，支持批量和实时评估。
    """
    
    def __init__(self, config: Optional[ModernEvaluatorConfig] = None):
        """
        初始化评估器
        
        Args:
            config: 评估器配置，如果为None则使用默认配置
        """
        self.config = config or ModernEvaluatorConfig()
        
        # 设置日志
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # 评估组件
        self.ragas_llm = None
        self.ragas_embeddings = None
        self.metrics = []
        
        # 评估历史
        self.evaluation_history: List[BatchEvaluationResult] = []
        
        # 初始化评估组件
        self._initialize_evaluation_components()
    
    def _initialize_evaluation_components(self) -> bool:
        """
        初始化评估组件
        
        Returns:
            bool: 初始化是否成功
        """
        if not RAGAS_AVAILABLE or not LANGCHAIN_AVAILABLE:
            logger.error("RAGAS或LangChain库不可用，评估器功能受限")
            return False
        
        try:
            # 设置API配置
            self._setup_api_configuration()
            
            # 初始化LLM和嵌入模型
            self._initialize_models()
            
            # 初始化评估指标
            self._initialize_metrics()
            
            logger.info("现代评估器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"评估器初始化失败: {e}")
            return False
    
    def _setup_api_configuration(self):
        """设置API配置"""
        # DashScope配置
        if self.config.dashscope_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.dashscope_api_key
            os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            logger.debug("DashScope API配置成功")
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            # LLM模型 - 使用DashScope
            if self.config.dashscope_api_key:
                self.llm = ChatOpenAI(
                    model=self.config.llm_model,
                    openai_api_key=self.config.dashscope_api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                self.ragas_llm = LangchainLLMWrapper(self.llm)
                logger.debug(f"LLM模型初始化成功: {self.config.llm_model}")
            
            # 嵌入模型 - 使用SiliconFlow
            if self.config.siliconflow_api_key:
                self.embeddings = OpenAIEmbeddings(
                    model=self.config.embedding_model,
                    openai_api_key=self.config.siliconflow_api_key,
                    openai_api_base="https://api.siliconflow.cn/v1"
                )
                self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
                logger.debug(f"嵌入模型初始化成功: {self.config.embedding_model}")
                
        except Exception as e:
            logger.warning(f"模型初始化失败: {e}，将使用默认配置")
            # 回退到默认配置
            self.ragas_llm = None
            self.ragas_embeddings = None
    
    def _initialize_metrics(self):
        """初始化评估指标"""
        self.metrics = []
        
        # 根据配置启用相应指标
        try:
            if self.config.enable_faithfulness:
                faithfulness = Faithfulness(llm=self.ragas_llm) if self.ragas_llm else Faithfulness()
                self.metrics.append(faithfulness)
                
            if self.config.enable_answer_relevancy:
                answer_relevancy = AnswerRelevancy(
                    llm=self.ragas_llm, 
                    embeddings=self.ragas_embeddings
                ) if (self.ragas_llm and self.ragas_embeddings) else AnswerRelevancy()
                self.metrics.append(answer_relevancy)
                
            if self.config.enable_context_precision:
                context_precision = ContextPrecision(llm=self.ragas_llm) if self.ragas_llm else ContextPrecision()
                self.metrics.append(context_precision)
                
            if self.config.enable_context_recall:
                context_recall = ContextRecall(llm=self.ragas_llm) if self.ragas_llm else ContextRecall()
                self.metrics.append(context_recall)
                
            if self.config.enable_semantic_similarity:
                semantic_similarity = SemanticSimilarity(
                    embeddings=self.ragas_embeddings
                ) if self.ragas_embeddings else SemanticSimilarity()
                self.metrics.append(semantic_similarity)
                
            if self.config.enable_answer_correctness:
                answer_correctness = AnswerCorrectness(
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings
                ) if (self.ragas_llm and self.ragas_embeddings) else AnswerCorrectness()
                self.metrics.append(answer_correctness)
                
            logger.info(f"评估指标初始化完成，共 {len(self.metrics)} 个指标")
            
        except Exception as e:
            logger.error(f"评估指标初始化失败: {e}")
            # 使用默认指标
            self.metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(), 
                ContextRecall(),
                SemanticSimilarity(),
                AnswerCorrectness()
            ]
    
    def is_available(self) -> bool:
        """检查评估器是否可用"""
        return (
            RAGAS_AVAILABLE and 
            LANGCHAIN_AVAILABLE and 
            len(self.metrics) > 0
        )
    
    async def evaluate_system(
        self,
        rag_system,
        test_questions: List[Dict[str, str]],
        system_name: str = "GraphRAG System"
    ) -> BatchEvaluationResult:
        """
        评估RAG系统
        
        Args:
            rag_system: RAG系统实例
            test_questions: 测试问题列表，格式为[{"question": "...", "ground_truth": "..."}, ...]
            system_name: 系统名称
            
        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        logger.info(f"开始评估系统: {system_name}")
        start_time = time.time()
        
        # 构建评估用例
        evaluation_cases = []
        for i, item in enumerate(test_questions):
            try:
                question = item["question"]
                ground_truth = item["ground_truth"]
                
                # 获取系统回答和上下文
                answer, contexts = await self._get_system_response(rag_system, question)
                
                case = EvaluationCase(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                    contexts=contexts,
                    metadata={"case_id": f"case_{i+1}", "original_data": item}
                )
                evaluation_cases.append(case)
                
                logger.debug(f"评估用例 {i+1} 准备完成")
                
                # API限制控制
                if i < len(test_questions) - 1:
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"评估用例 {i+1} 准备失败: {e}")
                # 添加空白用例，避免中断评估
                case = EvaluationCase(
                    question=item.get("question", ""),
                    ground_truth=item.get("ground_truth", ""),
                    answer="",
                    contexts=[],
                    metadata={"case_id": f"case_{i+1}", "error": str(e)}
                )
                evaluation_cases.append(case)
        
        # 执行批量评估
        batch_result = await self._execute_batch_evaluation(
            evaluation_cases, system_name, start_time
        )
        
        # 保存到评估历史
        self.evaluation_history.append(batch_result)
        
        logger.info(f"系统评估完成: {system_name}")
        return batch_result
    
    async def _get_system_response(
        self, 
        rag_system, 
        question: str
    ) -> Tuple[str, List[str]]:
        """
        获取RAG系统的回答和上下文
        
        Args:
            rag_system: RAG系统实例
            question: 问题
            
        Returns:
            Tuple[str, List[str]]: (回答, 上下文列表)
        """
        try:
            # 调用RAG系统获取回答
            answer = await rag_system.aquery(question)
            
            # 尝试获取上下文信息
            # 这里需要根据具体的RAG系统实现来调整
            contexts = self._extract_contexts_from_answer(answer)
            
            return answer, contexts
            
        except Exception as e:
            logger.error(f"获取系统回答失败: {e}")
            return "", []
    
    def _extract_contexts_from_answer(self, answer: str) -> List[str]:
        """
        从回答中提取上下文信息（简化版本）
        
        在未来的版本中，这应该通过RAG系统的API直接获取
        """
        if not answer or len(answer) < 50:
            return []
        
        # 简化策略：将回答作为上下文
        # 实际应用中应该从RAG系统获取原始检索结果
        context_chunks = []
        
        # 将长回答分割成较小的上下文块
        chunk_size = 200
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            if chunk.strip():
                context_chunks.append(chunk.strip())
        
        return context_chunks[:3]  # 最多返回3个上下文块
    
    async def _execute_batch_evaluation(
        self,
        evaluation_cases: List[EvaluationCase],
        system_name: str,
        start_time: float
    ) -> BatchEvaluationResult:
        """执行批量评估"""
        if not self.is_available():
            logger.error("评估器不可用，无法执行评估")
            return BatchEvaluationResult(
                system_name=system_name,
                total_cases=len(evaluation_cases),
                valid_cases=0,
                average_metrics={},
                individual_results=[],
                execution_time=time.time() - start_time,
                evaluation_config=asdict(self.config)
            )
        
        # 过滤有效用例
        valid_cases = [
            case for case in evaluation_cases 
            if case.answer and case.contexts
        ]
        
        logger.info(f"有效评估用例: {len(valid_cases)}/{len(evaluation_cases)}")
        
        if not valid_cases:
            logger.warning("没有有效的评估用例")
            return BatchEvaluationResult(
                system_name=system_name,
                total_cases=len(evaluation_cases),
                valid_cases=0,
                average_metrics={},
                individual_results=[],
                execution_time=time.time() - start_time,
                evaluation_config=asdict(self.config)
            )
        
        try:
            # 准备RAGAS数据格式
            ragas_data = {
                'question': [case.question for case in valid_cases],
                'answer': [case.answer for case in valid_cases], 
                'contexts': [case.contexts for case in valid_cases],
                'ground_truth': [case.ground_truth for case in valid_cases]
            }
            
            dataset = Dataset.from_dict(ragas_data)
            
            # 执行RAGAS评估
            logger.info("开始RAGAS评估...")
            ragas_result = evaluate(dataset, metrics=self.metrics)
            
            # 处理评估结果
            individual_results, average_metrics = self._process_ragas_results(
                ragas_result, valid_cases
            )
            
            execution_time = time.time() - start_time
            
            return BatchEvaluationResult(
                system_name=system_name,
                total_cases=len(evaluation_cases),
                valid_cases=len(valid_cases),
                average_metrics=average_metrics,
                individual_results=individual_results,
                execution_time=execution_time,
                evaluation_config=asdict(self.config)
            )
            
        except Exception as e:
            logger.error(f"RAGAS评估执行失败: {e}")
            # 返回错误结果
            return BatchEvaluationResult(
                system_name=system_name,
                total_cases=len(evaluation_cases),
                valid_cases=len(valid_cases),
                average_metrics={},
                individual_results=[],
                execution_time=time.time() - start_time,
                evaluation_config=asdict(self.config)
            )
    
    def _process_ragas_results(
        self, 
        ragas_result, 
        valid_cases: List[EvaluationCase]
    ) -> Tuple[List[EvaluationResult], Dict[str, float]]:
        """处理RAGAS评估结果"""
        individual_results = []
        average_metrics = {}
        
        try:
            # 转换RAGAS结果为DataFrame
            if hasattr(ragas_result, 'to_pandas'):
                df = ragas_result.to_pandas()
            else:
                logger.error("无法将RAGAS结果转换为DataFrame")
                return [], {}
            
            # 计算平均指标
            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                
                # 尝试多种可能的列名
                possible_columns = [
                    metric_name,
                    f"{metric_name}_score",
                    metric_name.replace('answercorrectness', 'answer_correctness'),
                    metric_name.replace('answerrelevancy', 'answer_relevancy'),
                    metric_name.replace('contextprecision', 'context_precision'),
                    metric_name.replace('contextrecall', 'context_recall'),
                    metric_name.replace('semanticsimilarity', 'semantic_similarity')
                ]
                
                for col in possible_columns:
                    if col in df.columns:
                        avg_score = float(df[col].mean())
                        average_metrics[metric_name] = avg_score
                        logger.debug(f"平均{metric_name}: {avg_score:.3f}")
                        break
                else:
                    logger.warning(f"未找到指标 {metric_name}，可用列: {list(df.columns)}")
                    average_metrics[metric_name] = 0.0
            
            # 构建个体结果
            for i, case in enumerate(valid_cases):
                if i < len(df):
                    case_metrics = {}
                    for metric_name in average_metrics:
                        # 查找对应的列
                        for col in df.columns:
                            if metric_name in col.lower():
                                case_metrics[metric_name] = float(df.iloc[i][col])
                                break
                        else:
                            case_metrics[metric_name] = 0.0
                    
                    result = EvaluationResult(
                        case_id=case.metadata.get("case_id", f"case_{i+1}"),
                        metrics=case_metrics,
                        raw_scores={},
                        execution_time=0.0,
                        metadata=case.metadata
                    )
                    individual_results.append(result)
            
            logger.info(f"处理了 {len(individual_results)} 个个体评估结果")
            
        except Exception as e:
            logger.error(f"RAGAS结果处理失败: {e}")
            # 返回空结果
            return [], {}
        
        return individual_results, average_metrics
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {"message": "暂无评估历史"}
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "latest_evaluation": asdict(self.evaluation_history[-1]) if self.evaluation_history else None,
            "average_metrics_trend": self._calculate_metrics_trend(),
            "config": asdict(self.config)
        }
        
        return summary
    
    def _calculate_metrics_trend(self) -> Dict[str, List[float]]:
        """计算指标趋势"""
        trends = {}
        
        for result in self.evaluation_history:
            for metric_name, score in result.average_metrics.items():
                if metric_name not in trends:
                    trends[metric_name] = []
                trends[metric_name].append(score)
        
        return trends
    
    def save_evaluation_results(
        self, 
        result: BatchEvaluationResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        保存评估结果
        
        Args:
            result: 评估结果
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            str: 保存的文件路径
        """
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_result_{timestamp}.json"
        
        # 准备保存数据
        save_data = {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_name": result.system_name,
                "evaluator_version": "1.0.0",
                "config": result.evaluation_config
            },
            "results": asdict(result)
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
            raise e

# 全局实例（可选）
_global_evaluator = None

def get_global_evaluator(config: Optional[ModernEvaluatorConfig] = None) -> ModernEvaluator:
    """获取全局评估器实例"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = ModernEvaluator(config)
    return _global_evaluator
