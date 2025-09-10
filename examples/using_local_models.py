"""
使用本地模型作为LLM和嵌入模型的示例

本示例展示如何使用本地模型（如Ollama、LM Studio等）作为LLM和嵌入模型。
包括：
- 配置本地模型
- 创建LLM和嵌入函数
- 使用本地模型进行RAG查询
"""

import os
import sys
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
import ollama
from typing import List, Dict, Any, Optional, Union
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger

# 配置日志
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# 模型配置
LLM_MODEL = "qwen2"  # Ollama模型名称
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama嵌入模型
EMBEDDING_MODEL_DIM = 768  # 嵌入维度
EMBEDDING_MODEL_MAX_TOKENS = 8192  # 最大token数

# 工作目录
WORKING_DIR = "./local_models_cache"


#------------------------------------------------------------------------------
# LLM函数
#------------------------------------------------------------------------------

async def ollama_llm(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: List[Dict[str, str]] = [], 
    **kwargs
) -> str:
    """使用Ollama作为LLM的函数"""
    # 移除Ollama不支持的参数
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    # 创建Ollama客户端
    ollama_client = ollama.AsyncClient()
    
    # 准备消息格式
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 缓存处理
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(LLM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # 调用Ollama API
    response = await ollama_client.chat(model=LLM_MODEL, messages=messages, **kwargs)
    result = response["message"]["content"]
    
    # 缓存响应
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": LLM_MODEL}})
    
    return result


#------------------------------------------------------------------------------
# 嵌入函数
#------------------------------------------------------------------------------

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS
)
async def ollama_embedding(texts: List[str]) -> np.ndarray:
    """使用Ollama生成嵌入向量的函数"""
    # 创建Ollama客户端
    ollama_client = ollama.Client()
    
    # 逐个处理文本
    embed_text = []
    for text in texts:
        try:
            data = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
            embed_text.append(data["embedding"])
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            # 添加零向量作为替代
            embed_text.append([0.0] * EMBEDDING_MODEL_DIM)
    
    return np.array(embed_text)


#------------------------------------------------------------------------------
# 辅助函数
#------------------------------------------------------------------------------

def remove_if_exist(file_path: str) -> None:
    """如果文件存在则删除"""
    if os.path.exists(file_path):
        os.remove(file_path)


def clean_cache_dir(cache_dir: str) -> None:
    """清理缓存目录"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # 删除缓存文件
    remove_if_exist(f"{cache_dir}/vdb_entities.json")
    remove_if_exist(f"{cache_dir}/kv_store_full_docs.json")
    remove_if_exist(f"{cache_dir}/kv_store_text_chunks.json")
    remove_if_exist(f"{cache_dir}/kv_store_community_reports.json")
    remove_if_exist(f"{cache_dir}/graph_chunk_entity_relation.graphml")


#------------------------------------------------------------------------------
# 示例函数
#------------------------------------------------------------------------------

def ollama_llm_only_example() -> None:
    """仅使用Ollama作为LLM的示例"""
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=f"{WORKING_DIR}/ollama_llm_only",
        best_model_func=ollama_llm,
        cheap_model_func=ollama_llm,
    )
    
    # 使用不同模式进行查询
    print("使用llm_only模式查询:")
    print(rag.query(
        "What is the capital of Australia?请用中文回复", 
        param=QueryParam(mode="llm_only")
    ))


def ollama_full_example() -> None:
    """同时使用Ollama作为LLM和嵌入模型的示例"""
    # 清理缓存目录
    cache_dir = f"{WORKING_DIR}/ollama_full"
    clean_cache_dir(cache_dir)
    
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=cache_dir,
        enable_llm_cache=True,
        best_model_func=ollama_llm,
        cheap_model_func=ollama_llm,
        embedding_func=ollama_embedding,
    )
    
    # 插入文档
    test_file = "./tests/mock_data.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        import time
        start = time.time()
        rag.insert(content)
        print(f"索引时间: {time.time() - start:.2f}秒")
        
        # 查询示例
        print("\n使用global模式查询:")
        print(rag.query(
            "What are the top themes in this story?请用中文回复", 
            param=QueryParam(mode="global")
        ))
    else:
        print(f"测试文件不存在: {test_file}")


if __name__ == "__main__":
    # 创建工作目录
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # 运行示例
    print("=== Ollama LLM示例 ===")
    ollama_llm_only_example()
    
    print("\n=== Ollama LLM + 嵌入示例 ===")
    ollama_full_example() 