"""
FAISS向量存储实现

提供基于Facebook FAISS库的高性能向量检索功能。
支持多种索引类型和批量操作。
"""

import os
import asyncio
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import faiss
import xxhash

from ...base import BaseVectorStorage
from ..._utils import logger


@dataclass
class FAISSVectorStorage(BaseVectorStorage):
    """FAISS向量存储实现
    
    Args:
        index_type: 索引类型 ("Flat", "IVFFlat", "HNSW", "PQ")
        nlist: IVF索引的聚类中心数量
        m: PQ量化的子向量数量
        nprobe: 搜索时的聚类数量
        metric: 距离度量 ("IP"内积, "L2"欧氏距离)
    """
    
    index_type: str = "Flat"
    nlist: int = 100
    m: int = 8
    nprobe: int = 10
    metric: str = "IP"  # 内积 (适合归一化向量)
    
    # 内部状态
    _index: Optional[faiss.Index] = field(default=None, init=False)
    _metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict, init=False)
    _index_file_name: str = field(default="", init=False)
    _metadata_file_name: str = field(default="", init=False)
    _max_batch_size: int = field(default=100, init=False)

    def __post_init__(self):
        """初始化FAISS索引"""
        # 修复：BaseVectorStorage没有__post_init__方法，直接初始化
        
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], 
            f"{self.namespace}_faiss_{self.index_type.lower()}.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], 
            f"{self.namespace}_metadata.pkl"
        )
        self._max_batch_size = self.global_config.get("embedding_batch_num", 100)
        
        # 尝试加载现有索引
        if self._load_existing_index():
            logger.info(f"已加载现有FAISS索引: {self._index_file_name}")
        else:
            self._create_new_index()
            logger.info(f"创建新的FAISS索引: {self.index_type}")

    def _load_existing_index(self) -> bool:
        """加载现有索引"""
        try:
            if os.path.exists(self._index_file_name) and os.path.exists(self._metadata_file_name):
                self._index = faiss.read_index(self._index_file_name)
                with open(self._metadata_file_name, 'rb') as f:
                    self._metadata = pickle.load(f)
                return True
        except Exception as e:
            logger.warning(f"加载现有索引失败: {e}")
        return False

    def _create_new_index(self):
        """创建新索引"""
        embedding_dim = self.embedding_func.embedding_dim
        
        if self.index_type == "Flat":
            if self.metric == "IP":
                base_index = faiss.IndexFlatIP(embedding_dim)
            else:
                base_index = faiss.IndexFlatL2(embedding_dim)
                
        elif self.index_type == "IVFFlat":
            if self.metric == "IP":
                quantizer = faiss.IndexFlatIP(embedding_dim)
                base_index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist)
            else:
                quantizer = faiss.IndexFlatL2(embedding_dim)
                base_index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist)
            base_index.nprobe = self.nprobe
            
        elif self.index_type == "HNSW":
            base_index = faiss.IndexHNSWFlat(embedding_dim, 32)
            base_index.hnsw.efConstruction = 200
            base_index.hnsw.efSearch = 50
            
        elif self.index_type == "PQ":
            base_index = faiss.IndexPQ(embedding_dim, self.m, 8)
            
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # 使用IndexIDMap包装以支持自定义ID
        self._index = faiss.IndexIDMap(base_index)
        self._metadata = {}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> int:
        """插入或更新向量数据
        
        Args:
            data: 键值对数据，键为ID，值包含content字段
            
        Returns:
            插入的向量数量
        """
        if not data:
            return 0
            
        logger.info(f"向{self.namespace}插入{len(data)}个向量")
        
        # 提取内容并分批处理
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        
        # 并行计算嵌入
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list).astype(np.float32)
        
        # 归一化向量（如果使用内积度量）
        if self.metric == "IP":
            faiss.normalize_L2(embeddings)
        
        # 准备ID和元数据
        ids = []
        for k, v in data.items():
            # 使用内容哈希作为ID
            doc_id = xxhash.xxh32_intdigest(k.encode())
            
            # 存储元数据
            metadata = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            metadata['id'] = k
            metadata['content'] = v.get('content', '')
            self._metadata[doc_id] = metadata
            
            ids.append(doc_id)
        
        # 添加到索引
        ids = np.array(ids, dtype=np.int64)
        
        # 如果是IVF索引且未训练，需要先训练
        if hasattr(self._index.index, 'is_trained') and not self._index.index.is_trained:
            logger.info("训练IVF索引...")
            self._index.index.train(embeddings)
        
        self._index.add_with_ids(embeddings, ids)
        
        logger.info(f"成功插入{len(data)}个向量，总数量: {self._index.ntotal}")
        return len(data)

    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """查询相似向量
        
        Args:
            query: 查询文本
            top_k: 返回的最相似结果数量
            
        Returns:
            相似结果列表，包含距离和元数据
        """
        if self._index.ntotal == 0:
            return []
        
        # 计算查询向量
        embedding = await self.embedding_func([query])
        embedding = np.array(embedding, dtype=np.float32)
        
        # 归一化查询向量
        if self.metric == "IP":
            faiss.normalize_L2(embedding)
        
        # 执行搜索
        distances, indices = self._index.search(embedding, min(top_k, self._index.ntotal))
        
        # 构建结果
        results = []
        for distance, doc_id in zip(distances[0], indices[0]):
            if doc_id != -1 and doc_id in self._metadata:  # -1表示空槽位
                metadata = self._metadata[doc_id].copy()
                
                # 计算相似度分数
                if self.metric == "IP":
                    metadata["similarity"] = float(distance)  # 内积已经是相似度
                else:
                    metadata["similarity"] = 1.0 / (1.0 + float(distance))  # L2距离转相似度
                
                metadata["distance"] = float(distance)
                results.append(metadata)
        
        return results

    async def index_done_callback(self):
        """索引构建完成后的回调，保存索引到磁盘"""
        try:
            os.makedirs(os.path.dirname(self._index_file_name), exist_ok=True)
            
            # 保存索引
            faiss.write_index(self._index, self._index_file_name)
            
            # 保存元数据
            with open(self._metadata_file_name, 'wb') as f:
                pickle.dump(self._metadata, f)
            
            logger.info(f"FAISS索引已保存: {self._index_file_name}")
            
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")

    async def query_done_callback(self):
        """查询完成后的回调"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if self._index is None:
            return {}
        
        stats = {
            "index_type": self.index_type,
            "total_vectors": self._index.ntotal,
            "embedding_dim": self.embedding_func.embedding_dim,
            "metric": self.metric,
            "metadata_count": len(self._metadata)
        }
        
        # 添加特定索引类型的统计信息
        if hasattr(self._index.index, 'nlist'):
            stats["nlist"] = self._index.index.nlist
        if hasattr(self._index.index, 'nprobe'):
            stats["nprobe"] = self._index.index.nprobe
            
        return stats

    def optimize_index(self):
        """优化索引（如果支持）"""
        if hasattr(self._index.index, 'optimize'):
            logger.info("优化FAISS索引...")
            self._index.index.optimize()


# 便利函数
def create_faiss_storage(
    namespace: str,
    global_config: Dict[str, Any],
    embedding_func,
    index_type: str = "Flat",
    **kwargs
) -> FAISSVectorStorage:
    """创建FAISS向量存储实例
    
    Args:
        namespace: 命名空间
        global_config: 全局配置
        embedding_func: 嵌入函数
        index_type: 索引类型
        **kwargs: 其他配置参数
        
    Returns:
        FAISSVectorStorage实例
    """
    return FAISSVectorStorage(
        namespace=namespace,
        global_config=global_config,
        embedding_func=embedding_func,
        index_type=index_type,
        **kwargs
    ) 