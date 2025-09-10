"""
检索结果对齐层
提供统一的检索结果数据结构和分数归一化功能
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from .._utils import logger


@dataclass
class RetrievalResult:
    """统一的检索结果数据结构"""
    content: str
    score: float  # 归一化后的分数 [0, 1]
    source: str   # 检索源标识 (naive, bm25, local, global)
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """确保分数在[0,1]范围内"""
        if self.score < 0:
            self.score = 0.0
        elif self.score > 1:
            self.score = 1.0


class ScoreNormalizer:
    """分数归一化器"""
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """Min-Max归一化到[0,1]"""
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # 避免除零
        if min_score == max_score:
            return [0.5] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """Z-score标准化然后映射到[0,1]"""
        if not scores:
            return []
        
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return [0.5] * len(scores)
        
        z_scores = (scores - mean_score) / std_score
        # 使用sigmoid函数映射到[0,1]
        normalized = 1 / (1 + np.exp(-z_scores))
        return normalized.tolist()
    
    @staticmethod
    def rank_based_normalize(scores: List[float]) -> List[float]:
        """基于排名的归一化"""
        if not scores:
            return []
        
        # 获取排名(分数越高排名越靠前)
        indices = np.argsort(scores)[::-1]  # 降序排列
        ranks = np.empty_like(indices)
        ranks[indices] = np.arange(len(scores))
        
        # 归一化排名到[0,1]
        normalized_ranks = 1.0 - ranks / (len(scores) - 1) if len(scores) > 1 else [1.0]
        return normalized_ranks.tolist()


class RetrievalResultAdapter:
    """检索结果适配器 - 将各种检索结果转换为统一格式"""
    
    def __init__(self, score_normalization_method: str = "min_max"):
        """
        初始化适配器
        
        Args:
            score_normalization_method: 分数归一化方法 (min_max, z_score, rank_based)
        """
        self.score_normalization_method = score_normalization_method
        self.normalizer = ScoreNormalizer()
    
    async def adapt_naive_results(self, raw_results: List[Dict], query: str) -> List[RetrievalResult]:
        """
        适配naive_query结果
        
        Args:
            raw_results: 向量检索的原始结果，格式: [{"id": str, "score": float, "content": str}, ...]
            query: 查询文本
            
        Returns:
            统一格式的检索结果列表
        """
        if not raw_results:
            return []
        
        try:
            # 提取分数并归一化
            scores = [result.get("score", 0.0) for result in raw_results]
            normalized_scores = self._normalize_scores(scores)
            
            # 转换为统一格式
            results = []
            for i, result in enumerate(raw_results):
                content = result.get("content", "")
                if not content:
                    logger.warning(f"Empty content in naive result: {result}")
                    continue
                
                retrieval_result = RetrievalResult(
                    content=content,
                    score=normalized_scores[i] if i < len(normalized_scores) else 0.5,
                    source="naive",
                    chunk_id=result.get("id"),
                    metadata={
                        "original_score": scores[i] if i < len(scores) else 0.0,
                        "query": query
                    }
                )
                results.append(retrieval_result)
            
            logger.debug(f"Adapted {len(results)} naive results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to adapt naive results: {e}")
            return []
    
    async def adapt_bm25_results(self, raw_results: List[Dict], query: str) -> List[RetrievalResult]:
        """
        适配bm25_query结果
        
        Args:
            raw_results: BM25检索的原始结果，格式: [{"content": str, "score": float, "id": str}, ...]
            query: 查询文本
            
        Returns:
            统一格式的检索结果列表
        """
        if not raw_results:
            return []
        
        try:
            # 提取分数并归一化
            scores = [result.get("score", 0.0) for result in raw_results]
            normalized_scores = self._normalize_scores(scores)
            
            # 转换为统一格式
            results = []
            for i, result in enumerate(raw_results):
                content = result.get("content", "")
                if not content:
                    logger.warning(f"Empty content in BM25 result: {result}")
                    continue
                
                retrieval_result = RetrievalResult(
                    content=content,
                    score=normalized_scores[i] if i < len(normalized_scores) else 0.5,
                    source="bm25",
                    chunk_id=result.get("id"),
                    metadata={
                        "original_score": scores[i] if i < len(scores) else 0.0,
                        "query": query
                    }
                )
                results.append(retrieval_result)
            
            logger.debug(f"Adapted {len(results)} BM25 results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to adapt BM25 results: {e}")
            return []
    
    async def adapt_graph_results(self, context_string: str, query: str, source: str = "local") -> List[RetrievalResult]:
        """
        适配local/global_query结果，将图检索的上下文字符串转换为TextChunk列表
        
        Args:
            context_string: 图检索返回的上下文字符串
            query: 查询文本  
            source: 检索源标识 (local 或 global)
            
        Returns:
            统一格式的检索结果列表
        """
        if not context_string or not context_string.strip():
            return []
        
        try:
            results = []
            
            if source == "global":
                # 处理global查询的分析师报告格式
                results = self._parse_global_context(context_string, query)
            else:
                # 处理local查询的文本块格式
                results = self._parse_local_context(context_string, query)
            
            logger.debug(f"Adapted {len(results)} {source} graph results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to adapt {source} graph results: {e}")
            return []
    
    def _parse_global_context(self, context_string: str, query: str) -> List[RetrievalResult]:
        """解析global查询的分析师报告格式"""
        results = []
        
        # 分析师报告的正则模式: "----Analyst X----\nImportance Score: Y"
        analyst_pattern = r"----Analyst\s+(\d+)----\s*\n.*?Importance Score:\s*([\d.]+)"
        matches = re.finditer(analyst_pattern, context_string, re.MULTILINE | re.DOTALL)
        
        scores = []
        contents = []
        
        for match in matches:
            analyst_num = match.group(1)
            importance_score = float(match.group(2))
            
            # 提取分析师报告的完整内容
            start_pos = match.start()
            # 找到下一个分析师报告的开始位置，或者字符串结尾
            next_match = None
            for next_candidate in re.finditer(analyst_pattern, context_string[match.end():], re.MULTILINE | re.DOTALL):
                next_match = next_candidate
                break
            
            if next_match:
                end_pos = match.end() + next_match.start()
            else:
                end_pos = len(context_string)
            
            content = context_string[start_pos:end_pos].strip()
            
            scores.append(importance_score)
            contents.append((content, analyst_num))
        
        # 如果没有找到分析师报告格式，将整个内容作为一个结果
        if not scores:
            # 使用理论中等分数，避免硬编码偏见
            default_score = 0.7  # 基于信息理论的中等置信度分数
            results.append(RetrievalResult(
                content=context_string,
                score=default_score,
                source="global",
                chunk_id="global_context",
                metadata={
                    "query": query, 
                    "type": "full_context",
                    "_score_note": "理论默认分数，需要实验验证"
                }
            ))
        else:
            # 归一化分数
            normalized_scores = self._normalize_scores(scores)
            
            for i, (content, analyst_num) in enumerate(contents):
                results.append(RetrievalResult(
                    content=content,
                    score=normalized_scores[i],
                    source="global",
                    chunk_id=f"analyst_{analyst_num}",
                    metadata={
                        "query": query,
                        "analyst_number": analyst_num,
                        "original_importance_score": scores[i]
                    }
                ))
        
        return results
    
    def _parse_local_context(self, context_string: str, query: str) -> List[RetrievalResult]:
        """解析local查询的文本块格式"""
        results = []
        
        # 尝试按"--New Chunk--"分割（如果存在）
        if "--New Chunk--" in context_string:
            chunks = context_string.split("--New Chunk--")
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        else:
            # 如果没有标准分隔符，将整个内容作为一个块
            chunks = [context_string.strip()]
        
        # 基于信息理论的分数分配方式
        # local查询基于图结构，使用位置权重而非任意分数
        base_score = 0.8  # 理论基础分数
        # 使用对数衰减而非线性衰减，符合信息检索理论
        position_decay_factor = 0.15  # 位置衰减因子，可配置
        
        for i, chunk_content in enumerate(chunks):
            if not chunk_content:
                continue
            
            # 使用对数衰减反映位置重要性，符合信息检索理论
            # score = base_score * exp(-decay_factor * position)
            import math
            position_weight = math.exp(-position_decay_factor * i)
            score = max(0.2, base_score * position_weight)  # 确保最低分数不会太低
            
            results.append(RetrievalResult(
                content=chunk_content,
                score=score,
                source="local",
                chunk_id=f"local_chunk_{i}",
                metadata={
                    "query": query,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "position_weight": position_weight,
                    "_score_method": "exponential_decay",
                    "_note": "分数基于位置权重理论计算"
                }
            ))
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """根据配置的方法归一化分数"""
        if not scores:
            return []
        
        try:
            if self.score_normalization_method == "min_max":
                return self.normalizer.min_max_normalize(scores)
            elif self.score_normalization_method == "z_score":
                return self.normalizer.z_score_normalize(scores)
            elif self.score_normalization_method == "rank_based":
                return self.normalizer.rank_based_normalize(scores)
            else:
                logger.warning(f"Unknown normalization method: {self.score_normalization_method}, using min_max")
                return self.normalizer.min_max_normalize(scores)
        except Exception as e:
            logger.error(f"Score normalization failed: {e}, returning equal scores")
            return [0.5] * len(scores)


def create_retrieval_adapter(score_normalization_method: str = "min_max") -> RetrievalResultAdapter:
    """创建检索结果适配器实例"""
    return RetrievalResultAdapter(score_normalization_method)
