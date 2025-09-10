"""
复杂度感知路由器模块
基于查询复杂度自动选择最佳检索策略
"""

import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .classifier import ComplexityClassifier, ComplexityClassifierConfig
from .._utils import logger

class BaseRouter(ABC):
    """路由器基类 - 提供基本的路由接口"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    async def route(self, query: str, available_modes: List[str] = None) -> str:
        """路由查询到检索模式"""
        pass
    
    @abstractmethod
    def create_query_param(self, query: str, available_modes: List[str] = None, **kwargs) -> Any:
        """创建查询参数"""
        pass

@dataclass
class ComplexityAwareRouter(BaseRouter):
    """复杂度感知路由器
    
    基于ModernBERT复杂度分类器，将查询路由到最适合的检索模式。
    """
    
    model_path: str = "nano_graphrag/models/modernbert_complexity_classifier"
    confidence_threshold: float = 0.6
    enable_fallback: bool = True
    use_modernbert: bool = True
    
    def __post_init__(self):
        """初始化路由器"""
        super().__init__()
        
        # 复杂度到检索模式的映射
        self._complexity_to_candidate = {
            "zero_hop": ["llm_only"],              # 无检索：直接LLM回答
            "one_hop": ["naive", "bm25"],          # 单跳检索：向量检索或关键词检索
            "multi_hop": ["local", "global"],      # 多跳检索：图推理
        }
        
        # 初始化分类器
        self.classifier = None
        if self.use_modernbert:
            try:
                config = ComplexityClassifierConfig(
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold
                )
                self.classifier = ComplexityClassifier(config)
                logger.info(f"复杂度分类器初始化成功: {self.model_path}")
            except Exception as e:
                logger.warning(f"复杂度分类器初始化失败: {e}")
                self.classifier = None
        
        # 统计信息
        self._complexity_stats = {
            "zero_hop": 0,
            "one_hop": 0,
            "multi_hop": 0,
            "fallback": 0
        }
    
    async def predict_complexity_detailed(self, query: str) -> Dict[str, Any]:
        """预测查询复杂度（详细版本）"""
        try:
            if self.classifier and self.classifier.is_available():
                # 使用ModernBERT分类器
                complexity, confidence, probabilities = self.classifier.predict_with_confidence(query)
                
                # 更新统计
                self._complexity_stats[complexity] += 1
                
                # 获取候选模式
                candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
                
                return {
                    "complexity": complexity,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "candidate_modes": candidate_modes,
                    "method": "modernbert"
                }
            else:
                # 回退到规则分类
                return await self._rule_based_complexity(query)
                
        except Exception as e:
            logger.error(f"复杂度预测失败: {e}")
            return await self._rule_based_complexity(query)
    
    async def predict_complexity(self, query: str) -> str:
        """预测查询复杂度（简化版本）"""
        result = await self.predict_complexity_detailed(query)
        return result["complexity"]
    
    async def _rule_based_complexity(self, query: str) -> Dict[str, Any]:
        """基于规则的复杂度分类"""
        # 简单的规则分类
        query_lower = query.lower()
        
        # zero-hop 规则
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            if len(query.split()) <= 5:
                complexity = "zero_hop"
            else:
                complexity = "one_hop"
        # multi-hop 规则
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity", "how does", "why does"]):
            complexity = "multi_hop"
        else:
            complexity = "one_hop"
        
        self._complexity_stats[complexity] += 1
        self._complexity_stats["fallback"] += 1
        
        candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
        
        return {
            "complexity": complexity,
            "confidence": 0.5,  # 规则分类的置信度较低
            "probabilities": {},
            "candidate_modes": candidate_modes,
            "method": "rule_based"
        }
    
    async def route(self, query: str, available_modes: List[str] = None) -> str:
        """路由查询到最佳检索模式"""
        if not available_modes:
            available_modes = ["llm_only", "naive", "bm25", "local", "global"]
        
        # 预测复杂度
        complexity_result = await self.predict_complexity_detailed(query)
        complexity = complexity_result["complexity"]
        confidence = complexity_result["confidence"]
        
        # 获取候选模式
        candidate_modes = complexity_result["candidate_modes"]
        
        # 从候选模式中选择可用的模式
        available_candidates = [mode for mode in candidate_modes if mode in available_modes]
        
        if not available_candidates:
            # 如果没有可用的候选模式，使用第一个可用模式
            logger.warning(f"复杂度 {complexity} 的候选模式 {candidate_modes} 都不可用，使用 {available_modes[0]}")
            return available_modes[0]
        
        # 如果置信度低于阈值且启用了回退，使用规则分类
        if confidence < self.confidence_threshold and self.enable_fallback:
            logger.info(f"置信度 {confidence:.3f} 低于阈值 {self.confidence_threshold}，使用规则回退")
            fallback_result = await self._rule_based_complexity(query)
            fallback_candidates = [mode for mode in fallback_result["candidate_modes"] if mode in available_modes]
            if fallback_candidates:
                return fallback_candidates[0]
        
        # 返回第一个可用的候选模式
        selected_mode = available_candidates[0]
        logger.info(f"查询复杂度: {complexity}, 置信度: {confidence:.3f}, 选择模式: {selected_mode}")
        
        return selected_mode
    
    def create_query_param(self, query: str, available_modes: List[str] = None, **kwargs) -> Any:
        """创建查询参数"""
        from ..base import QueryParam
        
        # 同步路由 - 优先使用训练好的分类器
        try:
            # 首先尝试使用训练好的ModernBERT分类器
            if self.classifier and self.classifier.is_available():
                try:
                    # 使用同步方式调用分类器
                    complexity, confidence, probabilities = self.classifier.predict_with_confidence(query)
                    
                    # 更新统计
                    self._complexity_stats[complexity] += 1
                    
                    # 获取候选模式
                    candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
                    
                    logger.info(f"ModernBERT分类器预测 - 复杂度: {complexity}, 置信度: {confidence:.3f}")
                    
                except Exception as e:
                    logger.warning(f"ModernBERT分类器预测失败，使用规则分类: {e}")
                    complexity_result = self._rule_based_complexity_sync(query)
                    complexity = complexity_result["complexity"]
                    candidate_modes = complexity_result["candidate_modes"]
                    confidence = complexity_result["confidence"]
            else:
                # 使用规则分类
                complexity_result = self._rule_based_complexity_sync(query)
                complexity = complexity_result["complexity"]
                candidate_modes = complexity_result["candidate_modes"]
                confidence = complexity_result["confidence"]
            
            # 从候选模式中选择可用的模式
            if not available_modes:
                available_modes = ["llm_only", "naive", "bm25", "local", "global"]
            
            available_candidates = [mode for mode in candidate_modes if mode in available_modes]
            
            if not available_candidates:
                # 如果没有可用的候选模式，使用第一个可用模式
                logger.warning(f"复杂度 {complexity} 的候选模式 {candidate_modes} 都不可用，使用 {available_modes[0]}")
                selected_mode = available_modes[0]
            else:
                selected_mode = available_candidates[0]
                
            logger.info(f"同步路由 - 查询复杂度: {complexity}, 置信度: {confidence:.3f}, 选择模式: {selected_mode}")
            
        except Exception as e:
            logger.warning(f"同步路由失败，使用默认模式: {e}")
            selected_mode = "naive" if not available_modes else available_modes[0]
        
        return QueryParam(
            mode=selected_mode,
            **kwargs
        )
    
    def _rule_based_complexity_sync(self, query: str) -> Dict[str, Any]:
        """基于规则的复杂度分类（同步版本）"""
        # 简单的规则分类
        query_lower = query.lower()
        
        # zero-hop 规则
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            if len(query.split()) <= 5:
                complexity = "zero_hop"
            else:
                complexity = "one_hop"
        # multi-hop 规则
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity", "how does", "why does"]):
            complexity = "multi_hop"
        else:
            complexity = "one_hop"
        
        self._complexity_stats[complexity] += 1
        self._complexity_stats["fallback"] += 1
        
        candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
        
        return {
            "complexity": complexity,
            "confidence": 0.5,  # 规则分类的置信度较低
            "probabilities": {},
            "candidate_modes": candidate_modes,
            "method": "rule_based_sync"
        }
    
    def get_complexity_stats(self) -> Dict[str, int]:
        """获取复杂度统计信息"""
        return self._complexity_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self._complexity_stats = {
            "zero_hop": 0,
            "one_hop": 0,
            "multi_hop": 0,
            "fallback": 0
        }
    
    def get_retrieval_plan(self, complexity_result: Dict[str, Any], available_modes: List[str] = None) -> List[str]:
        """
        基于置信度和复杂度概率分布生成检索计划
        
        实现三层渐进式检索策略：
        - 策略A：高置信度单路径（confidence >= 0.9）
        - 策略B：中等置信度双路径（0.6 <= confidence < 0.9）
        - 策略C：低置信度多路径（confidence < 0.6）
        
        Args:
            complexity_result: 复杂度分析结果
            available_modes: 可用的检索模式
            
        Returns:
            选择的检索模式列表
        """
        if not available_modes:
            available_modes = ["llm_only", "naive", "bm25", "local", "global"]
        
        confidence = complexity_result.get("confidence", 0.5)
        probabilities = complexity_result.get("probabilities", {})
        
        if confidence >= 0.9:
            # 策略A：高置信度单路径
            return self._get_optimal_mode(complexity_result, available_modes)
        elif confidence >= 0.6:
            # 策略B：中等置信度双路径
            return self._get_dual_modes_robust(complexity_result, available_modes)
        else:
            # 策略C：低置信度多路径
            return self._get_multi_modes_with_global_strategy(complexity_result, available_modes)
    
    def _get_optimal_mode(self, complexity_result: Dict[str, Any], available_modes: List[str]) -> List[str]:
        """获取最优单一模式（策略A）"""
        complexity = complexity_result.get("complexity", "one_hop")
        candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
        
        # 从候选模式中选择第一个可用的模式
        for mode in candidate_modes:
            if mode in available_modes:
                logger.info(f"策略A - 高置信度单路径: {mode}")
                return [mode]
        
        # 如果没有可用的候选模式，返回第一个可用模式
        logger.warning(f"候选模式 {candidate_modes} 都不可用，使用 {available_modes[0]}")
        return [available_modes[0]] if available_modes else []
    
    def _get_dual_modes_robust(self, complexity_result: Dict[str, Any], available_modes: List[str]) -> List[str]:
        """
        鲁棒的双路径选择，确保选择不同类型的检索器（策略B）
        
        Args:
            complexity_result: 复杂度分析结果
            available_modes: 可用的检索模式
            
        Returns:
            选择的双路径检索模式列表
        """
        probabilities = complexity_result.get("probabilities", {})
        
        # 按概率排序复杂度类别
        sorted_complexities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        selected_modes = []
        used_retriever_types = set()
        
        for complexity, prob in sorted_complexities:
            candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
            
            # 选择与已选模式不同类型的检索器
            for mode in candidate_modes:
                if mode in available_modes:
                    retriever_type = self._get_retriever_type(mode)
                    if retriever_type not in used_retriever_types:
                        selected_modes.append(mode)
                        used_retriever_types.add(retriever_type)
                        break
            
            if len(selected_modes) >= 2:
                break
        
        # 后备策略：确保至少有两个不同的检索器
        if len(selected_modes) < 2:
            backup_modes = self._get_diverse_retriever_combination(available_modes)
            selected_modes.extend(backup_modes[:2-len(selected_modes)])
        
        logger.info(f"策略B - 中等置信度双路径: {selected_modes[:2]}")
        return selected_modes[:2]
    
    def _get_retriever_type(self, mode: str) -> str:
        """获取检索器类型用于多样性检查"""
        type_mapping = {
            "naive": "vector",
            "bm25": "keyword", 
            "local": "graph_local",
            "global": "graph_global",
            "llm_only": "generation"
        }
        return type_mapping.get(mode, "unknown")
    
    def _get_diverse_retriever_combination(self, available_modes: List[str]) -> List[str]:
        """获取多样化的检索器组合"""
        # 按优先级排序的多样化组合
        preferred_combinations = [
            ["naive", "bm25"],      # 向量 + 关键词
            ["naive", "local"],     # 向量 + 图
            ["bm25", "local"],      # 关键词 + 图
            ["local", "global"],    # 本地图 + 全局图
        ]
        
        for combination in preferred_combinations:
            available_combination = [mode for mode in combination if mode in available_modes]
            if len(available_combination) >= 2:
                return available_combination
        
        # 如果没有理想组合，返回前两个可用模式
        return available_modes[:2] if len(available_modes) >= 2 else available_modes
    
    def _get_multi_modes_with_global_strategy(self, complexity_result: Dict[str, Any], available_modes: List[str]) -> List[str]:
        """
        带global策略的多路径选择（策略C）
        
        Args:
            complexity_result: 复杂度分析结果
            available_modes: 可用的检索模式
            
        Returns:
            选择的多路径检索模式列表
        """
        # 基础三路径：向量、关键词、图
        base_modes = ["naive", "bm25", "local"]
        selected_modes = [mode for mode in base_modes if mode in available_modes]
        
        # Global检索决策
        should_use_global = self._should_trigger_global_retrieval(complexity_result)
        
        if should_use_global and "global" in available_modes:
            # 策略：用global替换local（避免图检索重复，因为global包含local信息）
            if "local" in selected_modes:
                selected_modes.remove("local")
            selected_modes.append("global")
        
        logger.info(f"策略C - 低置信度多路径: {selected_modes}")
        return selected_modes
    
    def _should_trigger_global_retrieval(self, complexity_result: Dict[str, Any], query: str = None) -> bool:
        """
        Global检索的明确触发策略
        
        Args:
            complexity_result: 复杂度分析结果
            query: 查询文本（可选的启发式检查）
            
        Returns:
            是否应该触发Global检索
        """
        probabilities = complexity_result.get("probabilities", {})
        multi_hop_prob = probabilities.get("multi_hop", 0)
        confidence = complexity_result.get("confidence", 0.5)
        
        # 触发条件（优先级递减）：
        # 1. 多跳问题概率很高且置信度合理
        if multi_hop_prob > 0.6 and confidence > 0.4:
            logger.debug("Global触发条件1: 多跳概率高且置信度合理")
            return True
        
        # 2. 高度不确定的复杂推理问题（所有类别概率都较平均）
        if probabilities and max(probabilities.values()) < 0.4 and confidence < 0.6:
            logger.debug("Global触发条件2: 高度不确定的复杂推理问题")
            return True
        
        # 3. 查询中包含全局性关键词（可选的启发式检查）
        if query:
            global_keywords = ["overall", "summary", "general", "across", "comprehensive"]
            if any(keyword in query.lower() for keyword in global_keywords):
                logger.debug("Global触发条件3: 包含全局性关键词")
                return True
        
        return False 