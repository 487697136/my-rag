"""
多源检索融合模块

实现基于RRF(Reciprocal Rank Fusion)的多源检索结果融合算法，
支持置信度感知的自适应权重调整。
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .alignment import RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """融合配置参数"""
    k: float = 60.0  # RRF平滑参数
    max_results: int = 20  # 最大融合结果数
    diversity_threshold: float = 0.85  # 内容相似度阈值，超过则认为重复
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 1.0,
        "vector": 1.0, 
        "local": 1.0,
        "global": 1.0
    })
    
    # 置信度感知权重调整参数
    confidence_aware: bool = True  # 是否启用置信度感知
    weight_adjustment_factor: float = 2.0  # 权重调整因子
    base_weight: float = 0.5  # 基础权重


class ConfidenceAwareFusion:
    """置信度感知的多源检索融合器"""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """
        初始化融合器
        
        Args:
            config: 融合配置参数
        """
        self.config = config or FusionConfig()
        self.fusion_stats = {
            "total_fusions": 0,
            "avg_sources_per_fusion": 0,
            "avg_results_per_source": 0
        }
    
    def fuse_results(
        self,
        results_by_source: Dict[str, List[RetrievalResult]],
        query_complexity: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        融合多源检索结果
        
        Args:
            results_by_source: 按检索源分组的结果 {source_name: [RetrievalResult]}
            query_complexity: 查询复杂度信息，包含complexity和confidence
            
        Returns:
            融合后的排序结果列表
        """
        if not results_by_source:
            logger.warning("没有检索结果可供融合")
            return []
        
        # 统计信息
        self.fusion_stats["total_fusions"] += 1
        source_count = len(results_by_source)
        total_results = sum(len(results) for results in results_by_source.values())
        self.fusion_stats["avg_sources_per_fusion"] = (
            (self.fusion_stats["avg_sources_per_fusion"] * (self.fusion_stats["total_fusions"] - 1) + source_count) 
            / self.fusion_stats["total_fusions"]
        )
        
        logger.info(f"开始融合 {source_count} 个检索源的 {total_results} 个结果")
        
        # 第1步：内容去重
        deduplicated_results = self._deduplicate_results(results_by_source)
        
        # 第2步：计算置信度感知权重
        source_weights = self._calculate_confidence_aware_weights(
            list(deduplicated_results.keys()), 
            query_complexity
        )
        
        # 第3步：RRF融合计算
        fused_results = self._compute_rrf_scores(deduplicated_results, source_weights)
        
        # 第4步：多样性优化
        final_results = self._apply_diversity_filter(fused_results)
        
        logger.info(f"融合完成，返回 {len(final_results)} 个结果")
        return final_results[:self.config.max_results]
    
    def _deduplicate_results(
        self, 
        results_by_source: Dict[str, List[RetrievalResult]]
    ) -> Dict[str, List[RetrievalResult]]:
        """
        去除重复内容
        
        Args:
            results_by_source: 原始结果字典
            
        Returns:
            去重后的结果字典
        """
        # 构建内容到结果的映射，保留最高分的版本
        content_to_best_result = {}
        
        for source, results in results_by_source.items():
            for result in results:
                content_key = self._normalize_content_for_comparison(result.content)
                
                if content_key not in content_to_best_result:
                    content_to_best_result[content_key] = result
                else:
                    # 比较分数，保留更高分的结果
                    existing_result = content_to_best_result[content_key]
                    if result.normalize_score() > existing_result.normalize_score():
                        content_to_best_result[content_key] = result
        
        # 重新按源分组
        deduplicated_by_source = defaultdict(list)
        for result in content_to_best_result.values():
            deduplicated_by_source[result.source].append(result)
        
        # 重新排序每个源的结果
        for source in deduplicated_by_source:
            deduplicated_by_source[source].sort(
                key=lambda x: x.normalize_score(), 
                reverse=True
            )
            # 更新排名
            for i, result in enumerate(deduplicated_by_source[source]):
                result.rank = i + 1
        
        logger.debug(f"去重前总结果数: {sum(len(results) for results in results_by_source.values())}")
        logger.debug(f"去重后总结果数: {len(content_to_best_result)}")
        
        return dict(deduplicated_by_source)
    
    def _normalize_content_for_comparison(self, content: str) -> str:
        """
        规范化内容用于相似度比较
        
        Args:
            content: 原始内容
            
        Returns:
            规范化后的内容
        """
        # 简单的文本规范化：去除多余空白，转小写
        normalized = " ".join(content.lower().split())
        # 截取前200字符进行比较（避免极长文本比较开销）
        return normalized[:200]
    
    def _calculate_confidence_aware_weights(
        self,
        sources: List[str],
        query_complexity: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        计算置信度感知的权重
        
        Args:
            sources: 参与融合的检索源列表
            query_complexity: 查询复杂度信息
            
        Returns:
            各源的权重字典
        """
        # 从配置获取基础权重
        base_weights = {source: self.config.source_weights.get(source, 1.0) for source in sources}
        
        if not self.config.confidence_aware or not query_complexity:
            logger.debug("使用基础权重（未启用置信度感知或缺少复杂度信息）")
            return base_weights
        
        complexity = query_complexity.get("complexity", "one_hop")
        confidence = query_complexity.get("confidence", 0.5)
        
        logger.info(f"应用置信度感知权重调整: 复杂度={complexity}, 置信度={confidence:.3f}")
        
        # 计算复杂度概率 p (0-1范围，越高表示越复杂)
        if complexity == "zero_hop":
            complexity_prob = 0.1
        elif complexity == "one_hop":
            complexity_prob = 0.5
        elif complexity == "multi_hop":
            complexity_prob = 0.9
        else:
            complexity_prob = 0.5  # 默认中等复杂度
        
        # 根据置信度调整复杂度概率
        adjusted_prob = complexity_prob * confidence + 0.5 * (1 - confidence)
        
        # 计算自适应权重
        adjusted_weights = {}
        adjustment_factor = self.config.weight_adjustment_factor
        base_weight = self.config.base_weight
        
        for source in sources:
            if source == "global":
                # 全局图检索：复杂问题时权重增加
                weight = base_weight + adjustment_factor * adjusted_prob
            elif source == "bm25":
                # BM25检索：简单问题时权重增加
                weight = base_weight + adjustment_factor * (1 - adjusted_prob)
            elif source == "local":
                # 局部图检索：中等偏复杂问题权重较高
                weight = base_weight + adjustment_factor * (0.5 + 0.3 * adjusted_prob)
            elif source == "vector":
                # 向量检索：保持相对稳定，轻微偏向简单问题
                weight = base_weight + adjustment_factor * (0.3 + 0.2 * (1 - adjusted_prob))
            else:
                # 其他源使用基础权重
                weight = base_weights.get(source, 1.0)
            
            adjusted_weights[source] = max(0.1, weight)  # 确保权重不为0
        
        logger.debug(f"权重调整结果: {adjusted_weights}")
        return adjusted_weights
    
    def _compute_rrf_scores(
        self,
        results_by_source: Dict[str, List[RetrievalResult]],
        source_weights: Dict[str, float]
    ) -> List[RetrievalResult]:
        """
        计算RRF融合分数
        
        Args:
            results_by_source: 各源的结果
            source_weights: 各源的权重
            
        Returns:
            计算融合分数后的结果列表
        """
        # 收集所有独特结果
        all_results = {}  # result_id -> RetrievalResult
        for results in results_by_source.values():
            for result in results:
                if result.id not in all_results:
                    all_results[result.id] = result
        
        # 为每个结果计算RRF分数
        for result_id, result in all_results.items():
            rrf_score = 0.0
            
            # 查找该结果在各个源中的排名
            for source, results in results_by_source.items():
                source_weight = source_weights.get(source, 1.0)
                
                # 查找结果在该源中的排名
                rank_in_source = None
                for i, r in enumerate(results):
                    if r.id == result_id:
                        rank_in_source = i + 1
                        break
                
                if rank_in_source is not None:
                    # RRF公式: 1 / (k + rank)
                    rrf_contribution = source_weight / (self.config.k + rank_in_source)
                    rrf_score += rrf_contribution
            
            # 将RRF分数存储在metadata中
            result.metadata['rrf_score'] = rrf_score
            result.metadata['fusion_score'] = rrf_score  # 别名
        
        # 按RRF分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.metadata.get('rrf_score', 0.0),
            reverse=True
        )
        
        # 更新融合排名
        for i, result in enumerate(sorted_results):
            result.metadata['fusion_rank'] = i + 1
        
        logger.debug(f"RRF计算完成，排序 {len(sorted_results)} 个结果")
        return sorted_results
    
    def _apply_diversity_filter(
        self, 
        fused_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        应用多样性过滤，确保结果来源多样化
        
        Args:
            fused_results: RRF排序后的结果
            
        Returns:
            多样性过滤后的结果
        """
        if len(fused_results) <= 5:
            return fused_results
        
        filtered_results = []
        source_counts = defaultdict(int)
        
        for result in fused_results:
            source = result.source
            source_count = source_counts[source]
            
            # 限制每个源的最大结果数，确保多样性
            max_per_source = max(2, len(fused_results) // len(set(r.source for r in fused_results)))
            
            if source_count < max_per_source:
                filtered_results.append(result)
                source_counts[source] += 1
            elif len(filtered_results) < self.config.max_results // 2:
                # 如果总结果数还不够，允许适当放宽限制
                filtered_results.append(result)
                source_counts[source] += 1
        
        logger.debug(f"多样性过滤: {len(fused_results)} -> {len(filtered_results)}")
        logger.debug(f"各源分布: {dict(source_counts)}")
        
        return filtered_results
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        return self.fusion_stats.copy()


def create_fusion_engine(
    k: float = 60.0,
    max_results: int = 20,
    confidence_aware: bool = True,
    **kwargs
) -> ConfidenceAwareFusion:
    """
    创建融合引擎
    
    Args:
        k: RRF平滑参数
        max_results: 最大结果数
        confidence_aware: 是否启用置信度感知
        **kwargs: 其他配置参数
        
    Returns:
        融合引擎实例
    """
    config = FusionConfig(
        k=k,
        max_results=max_results,
        confidence_aware=confidence_aware,
        **kwargs
    )
    return ConfidenceAwareFusion(config)
