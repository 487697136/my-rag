"""
检索结果对齐与统一表示模块

提供统一的数据结构来表示不同检索源的结果，
便于后续融合处理。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import hashlib

@dataclass
class RetrievalResult:
    """
    统一的检索结果数据结构
    
    用于表示来自不同检索源的结果，包含内容、分数、来源等信息
    """
    content: str  # 检索到的文本内容
    score: float  # 检索相关性分数
    source: str   # 来源标识（如"bm25", "vector", "local", "global"）
    chunk_id: Optional[str] = None  # 文本块ID（如果有）
    rank: int = 1  # 在原始结果列表中的排名（1开始）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """后处理：生成唯一ID和规范化分数"""
        if not hasattr(self, '_id'):
            # 生成基于内容的唯一ID
            content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()[:8]
            self._id = f"{self.source}_{content_hash}"
    
    @property
    def id(self) -> str:
        """获取结果的唯一标识"""
        return self._id
    
    def normalize_score(self, method: str = "minmax") -> float:
        """
        归一化分数到0-1范围
        
        Args:
            method: 归一化方法，暂时保留接口
            
        Returns:
            归一化后的分数
        """
        # 对于不同来源的分数，采用不同的归一化策略
        if self.source == "bm25":
            # BM25分数通常范围较大，使用sigmoid归一化
            import math
            return 1 / (1 + math.exp(-self.score / 10))
        elif self.source in ["vector", "naive"]:
            # 向量相似度通常在0-1或-1-1范围
            if self.score < 0:
                return (self.score + 1) / 2  # 从[-1,1]映射到[0,1]
            return min(1.0, max(0.0, self.score))  # 确保在[0,1]范围内
        elif self.source in ["local", "global"]:
            # 图检索分数，通常已经是合理范围
            return min(1.0, max(0.0, self.score / 100))  # 假设原始分数0-100
        else:
            # 默认归一化
            return min(1.0, max(0.0, self.score))

class RetrievalAdapter:
    """检索结果适配器，将各种格式的检索结果转换为RetrievalResult格式"""
    
    async def adapt_naive_results(self, raw_results: List[Dict], query: str) -> List[RetrievalResult]:
        """适配naive/vector查询结果"""
        retrieval_results = []
        for i, result in enumerate(raw_results):
            retrieval_result = RetrievalResult(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                source="vector",  # naive查询使用vector作为源标识
                chunk_id=result.get("id"),
                rank=i + 1,
                metadata={"original_result": result}
            )
            retrieval_results.append(retrieval_result)
        return retrieval_results
    
    async def adapt_bm25_results(self, raw_results: List[Dict], query: str) -> List[RetrievalResult]:
        """适配BM25查询结果"""
        retrieval_results = []
        for i, result in enumerate(raw_results):
            retrieval_result = RetrievalResult(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                source="bm25",
                chunk_id=result.get("id"),
                rank=i + 1,
                metadata={"original_result": result}
            )
            retrieval_results.append(retrieval_result)
        return retrieval_results
    
    async def adapt_graph_results(self, context: str, query: str, source: str = "local") -> List[RetrievalResult]:
        """适配图检索结果（local/global）"""
        if not context or not context.strip():
            return []
        
        # 将图检索的上下文分割成独立的段落
        # 根据现有上下文格式进行分割
        sections = []
        
        # 尝试按照已知的分隔符分割
        if "-----Reports-----" in context:
            # Global查询格式
            parts = context.split("-----")
            for part in parts:
                part = part.strip()
                if part and not part.startswith("Reports") and not part.startswith("Entities") and not part.startswith("Relationships") and not part.startswith("Sources"):
                    if "```csv" in part:
                        # 提取CSV内容
                        csv_start = part.find("```csv\n") + 7
                        csv_end = part.find("\n```")
                        if csv_start != -1 and csv_end != -1:
                            csv_content = part[csv_start:csv_end].strip()
                            if csv_content:
                                sections.append(csv_content)
        elif "--New Chunk--" in context:
            # Naive查询格式
            sections = [s.strip() for s in context.split("--New Chunk--") if s.strip()]
        else:
            # 其他格式，直接使用整个上下文
            sections = [context.strip()]
        
        retrieval_results = []
        for i, section in enumerate(sections):
            if section and len(section.strip()) > 10:  # 过滤太短的段落
                retrieval_result = RetrievalResult(
                    content=section.strip(),
                    score=1.0 - (i * 0.1),  # 按顺序递减分数
                    source=source,
                    chunk_id=f"{source}_section_{i}",
                    rank=i + 1,
                    metadata={"context_section": True}
                )
                retrieval_results.append(retrieval_result)
        
        return retrieval_results


def create_retrieval_adapter() -> RetrievalAdapter:
    """创建检索结果适配器实例"""
    return RetrievalAdapter()


def align_retrieval_results(
    bm25_results: Optional[List] = None,
    vector_results: Optional[List] = None,
    local_results: Optional[List] = None,
    global_results: Optional[List] = None,
    **kwargs
) -> List[RetrievalResult]:
    """
    将不同检索源的结果对齐为统一格式
    
    Args:
        bm25_results: BM25检索结果
        vector_results: 向量检索结果  
        local_results: 局部图检索结果
        global_results: 全局图检索结果
        **kwargs: 其他检索结果
        
    Returns:
        统一格式的检索结果列表
    """
    aligned_results = []
    
    # 处理BM25结果
    if bm25_results:
        for i, result in enumerate(bm25_results):
            if hasattr(result, 'content') and hasattr(result, 'score'):
                # 已经是标准格式
                aligned_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    source="bm25",
                    chunk_id=getattr(result, 'chunk_id', None),
                    rank=i + 1,
                    metadata=getattr(result, 'metadata', {})
                )
            elif isinstance(result, dict):
                # 字典格式
                aligned_result = RetrievalResult(
                    content=result.get('content', str(result)),
                    score=result.get('score', 1.0),
                    source="bm25",
                    chunk_id=result.get('chunk_id'),
                    rank=i + 1,
                    metadata=result.get('metadata', {})
                )
            else:
                # 其他格式，尝试转换
                aligned_result = RetrievalResult(
                    content=str(result),
                    score=1.0,  # 默认分数
                    source="bm25",
                    rank=i + 1
                )
            aligned_results.append(aligned_result)
    
    # 处理向量检索结果
    if vector_results:
        for i, result in enumerate(vector_results):
            if hasattr(result, 'content') and hasattr(result, 'score'):
                aligned_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    source="vector",
                    chunk_id=getattr(result, 'chunk_id', None),
                    rank=i + 1,
                    metadata=getattr(result, 'metadata', {})
                )
            elif isinstance(result, dict):
                aligned_result = RetrievalResult(
                    content=result.get('content', str(result)),
                    score=result.get('score', 1.0),
                    source="vector", 
                    chunk_id=result.get('chunk_id'),
                    rank=i + 1,
                    metadata=result.get('metadata', {})
                )
            else:
                aligned_result = RetrievalResult(
                    content=str(result),
                    score=1.0,
                    source="vector",
                    rank=i + 1
                )
            aligned_results.append(aligned_result)
    
    # 处理局部图检索结果
    if local_results:
        for i, result in enumerate(local_results):
            if hasattr(result, 'content') and hasattr(result, 'score'):
                aligned_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    source="local",
                    chunk_id=getattr(result, 'chunk_id', None),
                    rank=i + 1,
                    metadata=getattr(result, 'metadata', {})
                )
            elif isinstance(result, dict):
                aligned_result = RetrievalResult(
                    content=result.get('content', str(result)),
                    score=result.get('score', 1.0),
                    source="local",
                    chunk_id=result.get('chunk_id'),
                    rank=i + 1,
                    metadata=result.get('metadata', {})
                )
            else:
                aligned_result = RetrievalResult(
                    content=str(result),
                    score=1.0,
                    source="local",
                    rank=i + 1
                )
            aligned_results.append(aligned_result)
    
    # 处理全局图检索结果
    if global_results:
        for i, result in enumerate(global_results):
            if hasattr(result, 'content') and hasattr(result, 'score'):
                aligned_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    source="global",
                    chunk_id=getattr(result, 'chunk_id', None),
                    rank=i + 1,
                    metadata=getattr(result, 'metadata', {})
                )
            elif isinstance(result, dict):
                aligned_result = RetrievalResult(
                    content=result.get('content', str(result)),
                    score=result.get('score', 1.0),
                    source="global",
                    chunk_id=result.get('chunk_id'),
                    rank=i + 1,
                    metadata=result.get('metadata', {})
                )
            else:
                aligned_result = RetrievalResult(
                    content=str(result),
                    score=1.0,
                    source="global",
                    rank=i + 1
                )
            aligned_results.append(aligned_result)
    
    # 处理其他检索结果
    for source_name, results in kwargs.items():
        if results:
            for i, result in enumerate(results):
                if hasattr(result, 'content') and hasattr(result, 'score'):
                    aligned_result = RetrievalResult(
                        content=result.content,
                        score=result.score,
                        source=source_name,
                        chunk_id=getattr(result, 'chunk_id', None),
                        rank=i + 1,
                        metadata=getattr(result, 'metadata', {})
                    )
                elif isinstance(result, dict):
                    aligned_result = RetrievalResult(
                        content=result.get('content', str(result)),
                        score=result.get('score', 1.0),
                        source=source_name,
                        chunk_id=result.get('chunk_id'),
                        rank=i + 1,
                        metadata=result.get('metadata', {})
                    )
                else:
                    aligned_result = RetrievalResult(
                        content=str(result),
                        score=1.0,
                        source=source_name,
                        rank=i + 1
                    )
                aligned_results.append(aligned_result)
    
    return aligned_results