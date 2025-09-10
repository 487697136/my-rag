import os
import re
import pickle
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
from collections import Counter
import math

from ...base import BaseKVStorage, StorageNameSpace
from ..._utils import logger


@dataclass
class BM25Storage(StorageNameSpace):
    """BM25存储类，用于BM25检索"""
    k1: float = 1.5
    b: float = 0.75
    
    def __post_init__(self):
        self._file_name = os.path.join(
            self.global_config["working_dir"], f"bm25_{self.namespace}.pkl"
        )
        self._index = {}  # 倒排索引: {token: {doc_id: freq}}
        self._doc_lengths = {}  # 文档长度: {doc_id: length}
        self._avg_doc_length = 0  # 平均文档长度
        self._documents = {}  # 文档内容: {doc_id: content}
        self._initialized = False
        
        if os.path.exists(self._file_name):
            try:
                with open(self._file_name, "rb") as f:
                    data = pickle.load(f)
                    self._index = data.get("index", {})
                    self._doc_lengths = data.get("doc_lengths", {})
                    self._avg_doc_length = data.get("avg_doc_length", 0)
                    self._documents = data.get("documents", {})
                    self._initialized = True
                logger.info(f"Loaded BM25 index for {self.namespace} with {len(self._documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词器，可以根据需要替换为更复杂的分词器"""
        # 去除特殊字符，转为小写
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 分词
        return text.split()
    
    def _calculate_idf(self, token: str) -> float:
        """计算逆文档频率 (IDF)"""
        if token not in self._index:
            return 0.0
        
        # 包含该词的文档数
        doc_count = len(self._index[token])
        # 总文档数
        total_docs = len(self._documents)
        
        # IDF计算公式
        return math.log((total_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)
    
    async def index_document(self, doc_id: str, content: str):
        """索引单个文档"""
        tokens = self._tokenize(content)
        doc_length = len(tokens)
        term_freqs = Counter(tokens)
        
        # 更新文档长度
        self._doc_lengths[doc_id] = doc_length
        
        # 更新倒排索引
        for token, freq in term_freqs.items():
            if token not in self._index:
                self._index[token] = {}
            self._index[token][doc_id] = freq
        
        # 更新文档内容
        self._documents[doc_id] = content
        
        # 更新平均文档长度
        self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths) if self._doc_lengths else 0
        
        self._initialized = True
    
    async def index_documents(self, documents: Dict[str, str]):
        """批量索引文档"""
        for doc_id, content in documents.items():
            await self.index_document(doc_id, content)
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """BM25搜索"""
        if not self._initialized:
            logger.warning("BM25 index not initialized")
            return []
        
        query_tokens = self._tokenize(query)
        scores = {}
        
        for token in query_tokens:
            if token not in self._index:
                continue
            
            idf = self._calculate_idf(token)
            
            for doc_id, term_freq in self._index[token].items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                
                doc_length = self._doc_lengths[doc_id]
                
                # BM25评分公式
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
                scores[doc_id] += idf * numerator / denominator
        
        # 按分数排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回结果
        results = []
        for doc_id, score in sorted_scores[:top_k]:
            results.append({
                "id": doc_id,
                "content": self._documents[doc_id],
                "score": score
            })
        
        return results
    
    async def index_start_callback(self):
        """开始索引回调"""
        pass
    
    async def index_done_callback(self):
        """索引完成回调"""
        with open(self._file_name, "wb") as f:
            data = {
                "index": self._index,
                "doc_lengths": self._doc_lengths,
                "avg_doc_length": self._avg_doc_length,
                "documents": self._documents
            }
            pickle.dump(data, f)
    
    async def query_done_callback(self):
        """查询完成回调"""
        pass 