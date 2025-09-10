#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NanoGraphRAG使用示例

演示如何初始化和使用NanoGraphRAG系统
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_graphrag.integration import NanoGraphRAG, NanoGraphRAGConfig
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("using_nanographrag")


async def init_with_default_config():
    """使用默认配置初始化NanoGraphRAG"""
    logger.info("===== 使用默认配置初始化NanoGraphRAG =====")
    
    # 加载嵌入函数和LLM函数
    import os
    
    # 获取API密钥
    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        logger.error("未设置DASHSCOPE_API_KEY环境变量")
        return None
    
    # 创建嵌入函数
    @wrap_embedding_func_with_attrs("bge", "dashscope")
    async def get_embedding_async(text):
        """
        获取文本嵌入向量
        """
        from dashscope.embeddings import AsyncEmbeddings
        
        try:
            response = await AsyncEmbeddings.call(
                model="text-embedding-v1",
                input=text,
                api_key=dashscope_api_key
            )
            
            if response.status_code == 200:
                return response.output["embeddings"][0]
            else:
                logger.error(f"获取嵌入向量失败: {response.code}, {response.message}")
                return None
        except Exception as e:
            logger.error(f"嵌入向量API调用异常: {str(e)}")
            return None
    
    # 创建LLM函数
    async def llm_func(prompt, **kwargs):
        """
        调用LLM API
        """
        from dashscope import Generation
        
        try:
            response = await Generation.async_call(
                model="qwen-turbo",
                prompt=prompt,
                api_key=dashscope_api_key,
                **kwargs
            )
            
            if response.status_code == 200:
                return response.output["text"]
            else:
                logger.error(f"LLM API调用失败: {response.code}, {response.message}")
                return f"API调用错误: {response.code}, {response.message}"
        except Exception as e:
            logger.error(f"LLM API调用异常: {str(e)}")
            return f"API调用异常: {str(e)}"
    
    # 创建配置
    config = NanoGraphRAGConfig(
        embedding_func=get_embedding_async,
        llm_func=llm_func,
        working_dir="./nanographrag_cache",
        api_type="dashscope",
        api_key=dashscope_api_key
    )
    
    # 初始化NanoGraphRAG
    rag = NanoGraphRAG(config)
    
    return rag


async def init_with_config_file():
    """使用配置文件初始化NanoGraphRAG"""
    logger.info("===== 使用配置文件初始化NanoGraphRAG =====")
    
    # 加载嵌入函数和LLM函数（与前面相同）
    # 这里省略实现，实际使用时需要提供
    import os
    
    # 获取API密钥
    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        logger.error("未设置DASHSCOPE_API_KEY环境变量")
        return None
    
    # 创建嵌入函数
    @wrap_embedding_func_with_attrs("bge", "dashscope")
    async def get_embedding_async(text):
        """获取文本嵌入向量"""
        from dashscope.embeddings import AsyncEmbeddings
        
        try:
            response = await AsyncEmbeddings.call(
                model="text-embedding-v1",
                input=text,
                api_key=dashscope_api_key
            )
            
            if response.status_code == 200:
                return response.output["embeddings"][0]
            else:
                logger.error(f"获取嵌入向量失败: {response.code}, {response.message}")
                return None
        except Exception as e:
            logger.error(f"嵌入向量API调用异常: {str(e)}")
            return None
    
    # 创建LLM函数
    async def llm_func(prompt, **kwargs):
        """调用LLM API"""
        from dashscope import Generation
        
        try:
            response = await Generation.async_call(
                model="qwen-turbo",
                prompt=prompt,
                api_key=dashscope_api_key,
                **kwargs
            )
            
            if response.status_code == 200:
                return response.output["text"]
            else:
                logger.error(f"LLM API调用失败: {response.code}, {response.message}")
                return f"API调用错误: {response.code}, {response.message}"
        except Exception as e:
            logger.error(f"LLM API调用异常: {str(e)}")
            return f"API调用异常: {str(e)}"
    
    # 方法1: 使用自定义配置文件
    config_path = "nano_graphrag/config/config_template.json"
    if os.path.exists(config_path):
        # 创建配置
        config = NanoGraphRAGConfig(
            config_path=config_path,
            embedding_func=get_embedding_async,
            llm_func=llm_func
        )
        
        # 初始化NanoGraphRAG
        rag = NanoGraphRAG(config)
    else:
        # 方法2: 使用默认配置模板
        config = NanoGraphRAGConfig.from_default_template(
            embedding_func=get_embedding_async,
            llm_func=llm_func,
            override_config={
                "working_dir": "./custom_cache",
                "api_key": dashscope_api_key,
                "api_type": "dashscope"
            }
        )
        
        # 初始化NanoGraphRAG
        rag = NanoGraphRAG(config)
    
    return rag


async def process_queries(rag: NanoGraphRAG, queries: List[str]):
    """处理多个查询"""
    for i, query in enumerate(queries):
        logger.info(f"处理查询 {i+1}/{len(queries)}: {query}")
        
        try:
            # 处理查询
            result = await rag.process_query(query)
            
            # 打印结果
            print(f"\n查询: {query}")
            print(f"复杂度: {result['complexity']}, 置信度: {result['confidence']:.2f}")
            print(f"检索模式: {result['retrieval_mode']}")
            print(f"检索到 {len(result['documents'])} 个文档")
            print(f"\n答案: {result['answer']}\n")
            print(f"耗时: {result['metadata']['elapsed_time']:.2f}秒\n")
            
            # 可选：添加反馈
            feedback_data = {
                "query": query,
                "user_rating": 5,  # 1-5分
                "user_comment": "答案很准确",
                "complexity": result['complexity']
            }
            rag.add_feedback(f"query_{hash(query)}", feedback_data)
            
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
    
    # 打印统计信息
    stats = rag.get_stats()
    print("\n系统统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


async def main():
    """主函数"""
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 选择初始化方法
    # 1. 使用默认配置
    rag = await init_with_default_config()
    
    if rag:
        # 测试查询
        try:
            result = await rag.process_query("什么是RAG技术？")
            logger.info(f"回答: {result['answer']}")
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
    else:
        logger.error("使用默认配置初始化失败")
    
    # 2. 使用配置文件
    rag2 = await init_with_config_file()
    
    if rag2:
        # 测试查询
        try:
            result = await rag2.process_query("图神经网络有哪些应用？")
            logger.info(f"回答(配置文件): {result['answer']}")
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
    else:
        logger.error("使用配置文件初始化失败")
    
    # 3. 使用默认模板
    try:
        # 获取嵌入函数和LLM函数
        import os
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        
        if dashscope_api_key:
            # 创建嵌入函数和LLM函数（简单示例）
            from dashscope.embeddings import AsyncEmbeddings
            from dashscope import Generation
            
            @wrap_embedding_func_with_attrs("bge", "dashscope")
            async def get_embedding(text):
                response = await AsyncEmbeddings.call(
                    model="text-embedding-v1", 
                    input=text, 
                    api_key=dashscope_api_key
                )
                return response.output["embeddings"][0]
            
            async def get_llm_response(prompt, **kwargs):
                response = await Generation.async_call(
                    model="qwen-turbo",
                    prompt=prompt,
                    api_key=dashscope_api_key,
                    **kwargs
                )
                return response.output["text"]
                
            # 使用默认模板创建配置
            config = NanoGraphRAGConfig.from_default_template(
                embedding_func=get_embedding,
                llm_func=get_llm_response,
                override_config={
                    "working_dir": "./template_cache",
                    "api_key": dashscope_api_key
                }
            )
            
            # 初始化NanoGraphRAG
            rag3 = NanoGraphRAG(config)
            
            # 测试查询
            result = await rag3.process_query("介绍一下纳米材料的应用")
            logger.info(f"回答(默认模板): {result['answer']}")
        else:
            logger.error("未设置DASHSCOPE_API_KEY环境变量，无法测试默认模板")
    
    except Exception as e:
        logger.error(f"默认模板测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 