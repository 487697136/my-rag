"""
使用各种API服务作为LLM和嵌入模型的示例

本示例展示如何使用不同的API服务（如SiliconFlow、阿里云百炼等）作为LLM和嵌入模型。
包括：
- 配置API密钥和基础URL
- 创建LLM和嵌入函数
- 处理API错误和重试
- 使用API服务进行RAG查询
"""

import os
import sys
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
import httpx
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 配置日志
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# API密钥和基础URL配置
# 硅基流动API
SILKFLOW_API_KEY = os.environ.get("SILKFLOW_API_KEY", "sk-rwcxtompyeenjkhdpuganvhsfmmctoftyfcqwpsgtchochkv")
SILKFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 阿里云百炼API
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
SILKFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 硅基流动模型
DASHSCOPE_MODEL = "qwen-turbo"               # 阿里云百炼模型
EMBED_MODEL = "BAAI/bge-m3"                  # 嵌入模型（1024维）

# 工作目录
WORKING_DIR = "./api_services_cache"


#------------------------------------------------------------------------------
# LLM API函数
#------------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError))
)
async def silkflow_llm_api(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: List[Dict[str, str]] = [], 
    **kwargs
) -> str:
    """调用硅基流动LLM API的函数"""
    # 准备消息格式
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 缓存处理
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(SILKFLOW_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # 准备请求头和请求体
    headers = {
        "Authorization": f"Bearer {SILKFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 从kwargs提取相关参数
    max_tokens = kwargs.pop("max_tokens", 512)
    temperature = kwargs.pop("temperature", 0.7)
    
    # 请求体
    payload = {
        "model": SILKFLOW_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # 可选参数
    if kwargs.get("stream", False):
        payload["stream"] = kwargs["stream"]
    
    try:
        # 发送请求
        async with httpx.AsyncClient(timeout=180.0) as client:
            logger.info(f"正在发送请求到 {SILKFLOW_API_BASE}/chat/completions")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求体: {payload}")
            
            response = await client.post(
                f"{SILKFLOW_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # 如果请求失败，记录错误详情
            if response.status_code != 200:
                logger.error(f"API请求失败 ({response.status_code}): {response.text}")
                response.raise_for_status()
                
            result = response.json()
            logger.debug(f"API响应: {result}")
        
        # 提取响应内容
        content = result["choices"][0]["message"]["content"]
        
        # 缓存响应
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": SILKFLOW_MODEL}}
            )
        
        return content
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error(f"API认证失败 (HTTP 401)：请检查您的API密钥是否正确")
        elif e.response.status_code == 400:
            logger.error(f"API请求格式错误 (HTTP 400)：{e.response.text}")
        elif e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 10))
            await asyncio.sleep(retry_after)      # 等待再重试
            raise
        else:
            logger.error(f"API请求失败 ({e.response.status_code}): {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"API请求过程中发生错误: {str(e)}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError))
)
async def dashscope_llm_api(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: List[Dict[str, str]] = [], 
    **kwargs
) -> str:
    """调用阿里云百炼LLM API的函数"""
    # 准备消息格式
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 缓存处理
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(DASHSCOPE_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # 准备请求头和请求体
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 从kwargs提取相关参数
    max_tokens = kwargs.pop("max_tokens", 512)
    temperature = kwargs.pop("temperature", 0.7)
    
    # 请求体
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # 可选参数
    if kwargs.get("stream", False):
        payload["stream"] = kwargs["stream"]
    
    try:
        # 发送请求
        async with httpx.AsyncClient(timeout=180.0) as client:
            logger.info(f"正在发送请求到 {DASHSCOPE_API_BASE}/chat/completions")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求体: {payload}")
            
            response = await client.post(
                f"{DASHSCOPE_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # 如果请求失败，记录错误详情
            if response.status_code != 200:
                logger.error(f"API请求失败 ({response.status_code}): {response.text}")
                response.raise_for_status()
                
            result = response.json()
            logger.debug(f"API响应: {result}")
        
        # 提取响应内容
        content = result["choices"][0]["message"]["content"]
        
        # 缓存响应
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": DASHSCOPE_MODEL}}
            )
        
        return content
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error(f"API认证失败 (HTTP 401)：请检查您的API密钥是否正确")
        elif e.response.status_code == 400:
            logger.error(f"API请求格式错误 (HTTP 400)：{e.response.text}")
        elif e.response.status_code == 429:
            logger.error(f"API请求过多 (HTTP 429)：请降低请求频率")
        else:
            logger.error(f"API请求失败 ({e.response.status_code}): {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"API请求过程中发生错误: {str(e)}")
        raise


#------------------------------------------------------------------------------
# 嵌入API函数
#------------------------------------------------------------------------------

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=4096)  # bge-m3通常是1024维
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError))
)
async def silkflow_embedding(texts: List[str]) -> np.ndarray:
    """调用硅基流动嵌入API的函数"""
    
    # 使用硅基流动API密钥
    headers = {
        "Authorization": f"Bearer {SILKFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    
    # 尝试批量处理所有文本，而不是一个一个处理
    try:
        payload = {
            "model": EMBED_MODEL,
            "input": texts,
            "encoding_format": "float"
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"正在发送嵌入请求到 {SILKFLOW_API_BASE}/embeddings")
            logger.debug(f"请求体: {payload}")
            
            response = await client.post(
                f"{SILKFLOW_API_BASE}/embeddings",
                headers=headers,
                json=payload
            )
            
            # 如果请求失败，记录错误详情
            if response.status_code != 200:
                logger.error(f"嵌入API请求失败 ({response.status_code}): {response.text}")
                response.raise_for_status()
                
            result = response.json()
            logger.debug(f"嵌入API响应: {result}")
            
            # 获取所有嵌入向量
            all_embeddings = [item["embedding"] for item in result["data"]]
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error(f"嵌入API认证失败 (HTTP 401)：请检查您的API密钥是否正确")
        elif e.response.status_code == 400:
            logger.error(f"嵌入API请求格式错误 (HTTP 400)：{e.response.text}")
            
            # 如果批量请求失败，尝试逐个请求
            logger.info("尝试逐个发送嵌入请求...")
            for text in texts:
                try:
                    payload = {
                        "model": EMBED_MODEL,
                        "input": text,
                        "encoding_format": "float"
                    }
                    
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{SILKFLOW_API_BASE}/embeddings",
                            headers=headers,
                            json=payload
                        )
                        response.raise_for_status()
                        result = response.json()
                        embedding = result["data"][0]["embedding"]
                        all_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"单个嵌入请求失败: {str(e)}")
                    # 添加零向量作为替代
                    all_embeddings.append([0.0] * 1024)
        else:
            logger.error(f"嵌入API请求失败 ({e.response.status_code}): {e.response.text}")
        if not all_embeddings:
            raise
    except Exception as e:
        logger.error(f"嵌入API请求过程中发生错误: {str(e)}")
        raise
    
    return np.array(all_embeddings)


#------------------------------------------------------------------------------
# 辅助函数
#------------------------------------------------------------------------------

def remove_if_exist(file_path: str) -> None:
    """如果文件存在则删除"""
    if os.path.exists(file_path):
        os.remove(file_path)


def load_json_data(json_dir: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """加载JSON数据文件"""
    import glob
    
    data = []
    for file_path in glob.glob(os.path.join(json_dir, pattern)):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL格式，每行一个JSON对象
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                else:
                    # 普通JSON格式
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {e}")
    
    return data


#------------------------------------------------------------------------------
# 示例函数
#------------------------------------------------------------------------------

def silkflow_example() -> None:
    """使用硅基流动API的示例"""
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=f"{WORKING_DIR}/silkflow",
        enable_naive_rag=True,
        best_model_func=silkflow_llm_api,
        cheap_model_func=silkflow_llm_api,
        embedding_func=silkflow_embedding,
    )
    
    # 使用不同模式进行查询
    print("使用naive模式查询:")
    print(rag.query(
        "What is the capital of Australia?请用中文回复", 
        param=QueryParam(mode="naive")
    ))
    
    print("\n使用llm_only模式查询:")
    print(rag.query(
        "What is the capital of Australia?请用中文回复", 
        param=QueryParam(mode="local")
    ))


def dashscope_example() -> None:
    """使用阿里云百炼API的示例"""
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=f"{WORKING_DIR}/dashscope",
        enable_naive_rag=True,
        best_model_func=dashscope_llm_api,
        cheap_model_func=dashscope_llm_api,
        embedding_func=silkflow_embedding,  # 仍使用硅基流动的嵌入API
    )
    
    # 使用不同模式进行查询
    print("使用naive模式查询:")
    print(rag.query(
        "What is the capital of Australia?请用中文回复", 
        param=QueryParam(mode="naive")
    ))
    
    print("\n使用llm_only模式查询:")
    print(rag.query(
        "What is the capital of Australia?请用中文回复", 
        param=QueryParam(mode="llm_only")
    ))


def insert_example(file_path: str) -> None:
    """插入文档示例"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # 清理旧缓存文件
    cache_dir = f"{WORKING_DIR}/insert_example"
    os.makedirs(cache_dir, exist_ok=True)
    
    remove_if_exist(f"{cache_dir}/vdb_entities.json")
    remove_if_exist(f"{cache_dir}/kv_store_full_docs.json")
    remove_if_exist(f"{cache_dir}/kv_store_text_chunks.json")
    remove_if_exist(f"{cache_dir}/kv_store_community_reports.json")
    remove_if_exist(f"{cache_dir}/graph_chunk_entity_relation.graphml")
    
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=cache_dir,
        enable_naive_rag=True,
        best_model_func=silkflow_llm_api,
        cheap_model_func=silkflow_llm_api,
        embedding_func=silkflow_embedding,
    )
    
    # 插入文档
    import time
    start = time.time()
    rag.insert(content)
    print(f"索引时间: {time.time() - start:.2f}秒")
    
    # 查询示例
    print("\n查询示例:")
    print(rag.query(
        "这篇文档主要讲了什么?请用中文回复", 
        param=QueryParam(mode="local")
    ))
    print("\n查询示例:")
    print(rag.query(
        "这篇文档主要讲了什么?请用中文回复", 
        param=QueryParam(mode="global")
    ))

if __name__ == "__main__":
    # 创建工作目录
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # 运行示例
    print("=== 硅基流动API示例 ===")
    silkflow_example()
    
    print("\n=== 阿里云百炼API示例 ===")
    dashscope_example()
    
    # 如果有测试文件，运行插入示例
    test_file = "./tests/zhuyuanzhang.txt"
    if os.path.exists(test_file):
        print("\n=== 文档插入示例 ===")
        insert_example(test_file) 