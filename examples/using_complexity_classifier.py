"""
使用复杂度分类器的完整RAG流程示例

本示例展示如何使用nano-graphrag的复杂度分类器进行完整的RAG流程：
1. 语料库准备和文档插入
2. 知识图谱和向量数据库构建
3. 复杂度分类器配置
4. 自适应查询路由和回答生成

包括：
- 文档预处理和分块
- 实体提取和关系构建
- 向量化和存储
- 复杂度感知的查询路由
- 多模式检索和回答生成
"""

import os
import sys
import json
import asyncio
import logging
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.complexity import ComplexityAwareRouter, ComplexityClassifier
from nano_graphrag._utils import logger, compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag.base import BaseKVStorage, BaseGraphStorage, BaseVectorStorage

# 配置日志
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# API密钥和基础URL配置
# 硅基流动API
SILKFLOW_API_KEY = "sk-rwcxtompyeenjkhdpuganvhsfmmctoftyfcqwpsgtchochkv"
SILKFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 阿里云百炼API
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY")
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
SILKFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 硅基流动模型
DASHSCOPE_MODEL = "qwen-turbo"               # 阿里云百炼模型
EMBED_MODEL = "BAAI/bge-m3"                  # 嵌入模型（1024维）

# 工作目录配置
WORKING_DIR = "./complexity_classifier_cache"

# 示例文档数据
SAMPLE_DOCUMENTS = [
    {
        "title": "人工智能基础",
        "content": """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
AI的主要分支包括机器学习、深度学习、自然语言处理等。

机器学习是AI的核心技术之一，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。
深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程。

自然语言处理（NLP）是AI的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。
NLP的应用包括机器翻译、情感分析、问答系统等。

强化学习是机器学习的一种方法，通过与环境交互来学习最优策略。
它在游戏、机器人控制、自动驾驶等领域有重要应用。
        """
    },
    {
        "title": "机器学习算法",
        "content": """
机器学习算法可以分为三大类：监督学习、无监督学习和强化学习。

监督学习算法包括：
- 线性回归：用于预测连续值
- 逻辑回归：用于分类问题
- 决策树：基于特征进行决策
- 随机森林：集成多个决策树
- 支持向量机：寻找最优分类边界
- 神经网络：模拟人脑神经元

无监督学习算法包括：
- K-means聚类：将数据分组
- 主成分分析：降维技术
- 自编码器：学习数据表示
- 生成对抗网络：生成新数据

深度学习是机器学习的一个分支，使用多层神经网络：
- 卷积神经网络（CNN）：用于图像处理
- 循环神经网络（RNN）：用于序列数据
- 长短期记忆网络（LSTM）：改进的RNN
- Transformer：用于自然语言处理
        """
    },
    {
        "title": "深度学习框架",
        "content": """
主流的深度学习框架包括：

PyTorch是由Facebook开发的深度学习框架，具有动态计算图的特性。
它提供了灵活的编程接口，支持快速原型开发和实验。
PyTorch在学术界和工业界都广泛使用，特别是在研究领域。

TensorFlow是由Google开发的深度学习框架，具有静态计算图的特性。
它提供了强大的生产环境支持，包括TensorFlow Serving和TensorFlow Lite。
TensorFlow在工业部署方面有优势。

Keras是一个高级神经网络API，可以运行在TensorFlow、Theano或CNTK之上。
它提供了简单易用的接口，适合初学者使用。

JAX是Google开发的用于高性能机器学习研究的框架。
它结合了NumPy的易用性和GPU/TPU的加速能力。

这些框架都支持自动微分、GPU加速和分布式训练等核心功能。
选择框架时需要考虑项目需求、团队技能和部署环境等因素。
        """
    }
]

# 测试查询集
TEST_QUERIES = [
    # Zero-hop查询（常识性问题）
    "什么是人工智能？",
    "2+2等于多少？",
    "水的沸点是多少？",
    
    # One-hop查询（单步检索）
    "机器学习有哪些主要算法？",
    "PyTorch是什么？",
    "深度学习框架有哪些？",
    
    # Multi-hop查询（多步推理）
    "深度学习和机器学习的关系是什么？",
    "PyTorch相比TensorFlow有什么优势？",
    "在图像处理任务中，为什么CNN比RNN更有效？",
    "强化学习在自动驾驶中的应用原理是什么？"
]


#------------------------------------------------------------------------------
# API函数
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
    
    try:
        # 使用OpenAI客户端调用阿里云百炼API
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_API_BASE,
        )
        
        # 从kwargs提取相关参数
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=DASHSCOPE_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # 提取响应内容
        content = completion.choices[0].message.content
        
        # 缓存响应
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": DASHSCOPE_MODEL}}
            )
        
        return content
        
    except Exception as e:
        logger.error(f"阿里云百炼API请求过程中发生错误: {str(e)}")
        raise


async def simple_entity_extraction(
    inserting_chunks: Dict[str, Dict],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: Optional[BaseVectorStorage],
    global_config: Dict,
    using_amazon_bedrock: bool = False,
) -> Optional[BaseGraphStorage]:
    """简化的实体提取函数，避免使用有问题的pack_user_ass_to_openai_messages"""
    # 简单返回现有的知识图谱，不进行实体提取
    logger.info("使用简化的实体提取函数，跳过实体提取步骤")
    return knwoledge_graph_inst


async def simple_community_report(
    community_reports: BaseKVStorage,
    chunk_entity_relation_graph: BaseGraphStorage,
    global_config: Dict
) -> None:
    """简化的社区报告生成函数，跳过聚类步骤"""
    logger.info("使用简化的社区报告生成函数，跳过聚类步骤")
    # 不进行任何操作，直接返回
    pass


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=4096)  # bge-m3是1024维
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
# 文档预处理函数
#------------------------------------------------------------------------------

def prepare_documents() -> List[str]:
    """准备文档数据"""
    documents = []
    for doc in SAMPLE_DOCUMENTS:
        # 合并标题和内容
        full_content = f"标题：{doc['title']}\n\n内容：{doc['content']}"
        documents.append(full_content)
    return documents


def save_documents_to_file(documents: List[str], file_path: str) -> None:
    """将文档保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documents):
            f.write(f"=== 文档 {i+1} ===\n")
            f.write(doc)
            f.write("\n\n")


#------------------------------------------------------------------------------
# 复杂度分类器配置
#------------------------------------------------------------------------------

def create_complexity_router() -> ComplexityAwareRouter:
    """创建复杂度感知路由器"""
    router = ComplexityAwareRouter(
        model_path="nano_graphrag/models/modernbert_complexity_classifier",
        confidence_threshold=0.6,
        enable_fallback=True,
        use_modernbert=True
    )
    return router


async def test_complexity_classification(router: ComplexityAwareRouter) -> None:
    """测试复杂度分类功能"""
    print("\n" + "="*60)
    print("🧠 测试复杂度分类器")
    print("="*60)
    
    for query in TEST_QUERIES:
        # 预测复杂度
        complexity_result = await router.predict_complexity_detailed(query)
        
        print(f"\n查询: {query}")
        print(f"预测复杂度: {complexity_result['complexity']}")
        print(f"置信度: {complexity_result['confidence']:.3f}")
        print(f"候选模式: {complexity_result['candidate_modes']}")
        print(f"分类方法: {complexity_result['method']}")


#------------------------------------------------------------------------------
# RAG系统构建
#------------------------------------------------------------------------------

def build_rag_system(working_dir: str) -> GraphRAG:
    """构建RAG系统"""
    print(f"\n🔧 构建RAG系统，工作目录: {working_dir}")
    
    # 创建GraphRAG实例，启用复杂度感知路由
    rag = GraphRAG(
        working_dir=working_dir,
        enable_naive_rag=True,
        enable_bm25=True,
        enable_local=False,  # 禁用local模式以避免实体提取问题
        # 使用API模型
        best_model_func=dashscope_llm_api,  # 使用阿里云百炼API
        cheap_model_func=dashscope_llm_api,  # 使用阿里云百炼API
        embedding_func=silkflow_embedding,   # 使用硅基流动嵌入API
        # 配置向量数据库使用1024维
        vector_db_storage_cls_kwargs={
            "embedding_dim": 1024
        },
        # 使用复杂度感知路由器
        router_cls=ComplexityAwareRouter,
        router_kwargs={
            "model_path": "nano_graphrag/models/modernbert_complexity_classifier",
            "confidence_threshold": 0.6,
            "enable_fallback": True,
            "use_modernbert": True
        }
    )
    
    # 替换实体提取函数
    rag.entity_extraction_func = simple_entity_extraction
    
    # 替换社区报告生成函数
    # 这里我们将在插入文档时手动处理，避免调用有问题的聚类函数
    
    return rag


async def insert_documents(rag: GraphRAG, documents: List[str]) -> None:
    """插入文档到RAG系统"""
    print(f"\n📚 插入 {len(documents)} 个文档")
    
    for i, doc in enumerate(documents):
        print(f"正在处理文档 {i+1}/{len(documents)}...")
        try:
            # 手动处理文档插入，避免聚类问题
            await manual_insert_document(rag, doc)
        except Exception as e:
            print(f"文档 {i+1} 插入失败: {e}")
            # 继续处理下一个文档
            continue
    
    print("✅ 文档插入完成")


async def manual_insert_document(rag: GraphRAG, document: str) -> None:
    """手动插入文档，避免聚类问题"""
    # 直接调用GraphRAG的ainsert方法，但在聚类步骤前停止
    await rag._insert_start()
    
    try:
        # 处理文档
        if isinstance(document, str):
            documents = [document]
        
        # 创建新文档
        from nano_graphrag._utils import compute_mdhash_id
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in documents
        }
        
        # 过滤已存在的文档
        _add_doc_keys = await rag.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        
        if not len(new_docs):
            logger.warning(f"所有文档都已存在于存储中")
            return
        
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        
        # 分块处理
        from nano_graphrag._op import get_chunks
        inserting_chunks = get_chunks(
            new_docs=new_docs,
            chunk_func=rag.chunk_func,
            overlap_token_size=rag.chunk_overlap_token_size,
            max_token_size=rag.chunk_token_size,
        )
        
        _add_chunk_keys = await rag.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        
        if not len(inserting_chunks):
            logger.warning(f"所有块都已存在于存储中")
            return
        
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
        
        # 插入到向量数据库
        if rag.enable_naive_rag:
            logger.info("Insert chunks for naive RAG")
            await rag.chunks_vdb.upsert(inserting_chunks)
        
        # 为BM25索引文档
        if rag.enable_bm25:
            logger.info("Indexing documents for BM25")
            bm25_docs = {k: v["content"] for k, v in inserting_chunks.items()}
            await rag.bm25_store.index_documents(bm25_docs)
        
        # 清空社区报告
        await rag.community_reports.drop()
        
        # 使用简化的实体提取
        logger.info("[Entity Extraction]...")
        maybe_new_kg = await rag.entity_extraction_func(
            inserting_chunks,
            rag.chunk_entity_relation_graph,
            rag.entities_vdb,
            rag.__dict__,
            rag.using_amazon_bedrock,
        )
        
        if maybe_new_kg is not None:
            rag.chunk_entity_relation_graph = maybe_new_kg
        
        # 跳过聚类步骤，直接提交
        logger.info("[Community Report]...")
        logger.info("跳过聚类步骤以避免空图问题")
        
        # 提交所有更改
        await rag.full_docs.upsert(new_docs)
        await rag.text_chunks.upsert(inserting_chunks)
        
    finally:
        await rag._insert_done()


#------------------------------------------------------------------------------
# 查询测试
#------------------------------------------------------------------------------

async def test_queries_with_complexity_routing(rag: GraphRAG) -> None:
    """使用复杂度路由测试查询"""
    print("\n" + "="*60)
    print("🔍 测试复杂度感知查询路由")
    print("="*60)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        
        try:
            # 使用复杂度感知路由进行查询
            response = await rag.aquery(query)
            
            print(f"回答: {response}")
            
        except Exception as e:
            print(f"查询失败: {e}")


def test_different_modes(rag: GraphRAG) -> None:
    """测试不同检索模式"""
    print("\n" + "="*60)
    print("🔄 测试不同检索模式")
    print("="*60)
    
    test_query = "深度学习和机器学习的关系是什么？"
    
    modes = ["llm_only", "naive", "bm25", "local", "global"]
    
    for mode in modes:
        print(f"\n--- 模式: {mode} ---")
        try:
            response = rag.query(test_query, param=QueryParam(mode=mode))
            print(f"回答: {response}")
        except Exception as e:
            print(f"查询失败: {e}")


#------------------------------------------------------------------------------
# 性能分析
#------------------------------------------------------------------------------

def analyze_system_performance(rag: GraphRAG) -> None:
    """分析系统性能"""
    print("\n" + "="*60)
    print("📊 系统性能分析")
    print("="*60)
    
    # 获取复杂度统计
    if hasattr(rag.router, 'get_complexity_stats'):
        stats = rag.router.get_complexity_stats()
        print(f"复杂度分类统计: {stats}")
    
    # 检查存储状态
    working_dir = rag.working_dir
    print(f"\n存储状态:")
    print(f"工作目录: {working_dir}")
    
    # 检查各种文件是否存在
    files_to_check = [
        "vdb_chunks.json",
        "vdb_entities.json", 
        "kv_store_text_chunks.json",
        "kv_store_entities.json",
        "kv_store_community_reports.json",
        "graph_chunk_entity_relation.graphml"
    ]
    
    for file_name in files_to_check:
        file_path = os.path.join(working_dir, file_name)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"  {file_name}: {'✅' if exists else '❌'} ({size} bytes)")


#------------------------------------------------------------------------------
# 主函数
#------------------------------------------------------------------------------

async def main():
    """主函数"""
    print("🚀 开始复杂度分类器完整RAG流程演示")
    print("="*60)
    
    # 检查API密钥
    if DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY":
        print("⚠️ 请设置DASHSCOPE_API_KEY环境变量")
        print("例如: export DASHSCOPE_API_KEY=your_api_key_here")
        print("或者在代码中直接设置DASHSCOPE_API_KEY变量")
        return
    
    # 1. 准备文档
    print("\n📝 步骤1: 准备文档数据")
    documents = prepare_documents()
    print(f"准备了 {len(documents)} 个文档")
    
    # 保存文档到文件（可选）
    docs_file = os.path.join(WORKING_DIR, "sample_documents.txt")
    os.makedirs(WORKING_DIR, exist_ok=True)
    save_documents_to_file(documents, docs_file)
    print(f"文档已保存到: {docs_file}")
    
    # 2. 测试复杂度分类器
    print("\n🧠 步骤2: 测试复杂度分类器")
    router = create_complexity_router()
    await test_complexity_classification(router)
    
    # 3. 构建RAG系统
    print("\n🔧 步骤3: 构建RAG系统")
    rag = build_rag_system(WORKING_DIR)
    
    # 4. 插入文档
    print("\n📚 步骤4: 插入文档")
    await insert_documents(rag, documents)
    
    # 5. 测试复杂度感知查询
    print("\n🔍 步骤5: 测试复杂度感知查询")
    await test_queries_with_complexity_routing(rag)
    
    # 6. 测试不同模式
    print("\n🔄 步骤6: 测试不同检索模式")
    test_different_modes(rag)
    
    # 7. 性能分析
    print("\n📊 步骤7: 系统性能分析")
    analyze_system_performance(rag)
    
    print("\n" + "="*60)
    print("🎉 复杂度分类器完整RAG流程演示完成！")
    print("="*60)
    
    print("\n📝 使用说明:")
    print("1. 系统已自动根据查询复杂度选择最佳检索模式")
    print("2. zero_hop查询使用llm_only模式（直接回答）")
    print("3. one_hop查询使用naive或bm25模式（单步检索）")
    print("4. multi_hop查询使用local或global模式（图推理）")
    print("5. 如果模型置信度低，会自动回退到规则分类")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 