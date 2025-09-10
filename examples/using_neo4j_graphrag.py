#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Neo4j作为图存储的GraphRAG示例

本示例展示如何配置和使用Neo4j数据库作为nano_graphrag的图存储后端。

前置要求:
1. 安装Neo4j数据库 (本地安装或Docker)
2. 安装neo4j Python驱动: pip install neo4j
3. 确保Neo4j服务正在运行

参考文档:
- Neo4j Python Driver: https://neo4j.com/docs/getting-started/languages-guides/neo4j-python/
- Neo4j Docker: https://hub.docker.com/_/neo4j
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage.graph.neo4j import Neo4jStorage


def get_neo4j_config():
    """获取Neo4j配置
    
    支持多种配置方式:
    1. 环境变量 (推荐)
    2. 配置文件
    3. 直接配置
    """
    
    # 方式1: 从环境变量读取 (推荐用于生产环境)
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j+s://47aa7a59.databases.neo4j.io")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "1COh0KFBOJL12ByoSj82D0eoeXpFv8bWWvocts83m9s")
    
    # 方式2: 从配置文件读取 (可选)
    # 您可以根据需要实现配置文件读取逻辑
    
    # 方式3: 直接配置 (仅用于开发和测试)
    config = {
        "working_dir": "./neo4j_graphrag_cache",
        "enable_llm_cache": True,
        
        # Neo4j配置参数
        "addon_params": {
            "neo4j_url": neo4j_uri,
            "neo4j_auth": (neo4j_username, neo4j_password),
            "neo4j_database": "neo4j",  # 数据库名称
        },
        
        # 其他GraphRAG配置
        "best_model_max_token_size": 4000,
        "cheap_model_max_token_size": 2000,
        "embedding_batch_num": 32,
    }
    
    return config


def test_neo4j_connection():
    """测试Neo4j数据库连接"""
    try:
        from neo4j import GraphDatabase
        
        # 获取连接参数
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j+s://47aa7a59.databases.neo4j.io")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "1COh0KFBOJL12ByoSj82D0eoeXpFv8bWWvocts83m9s")
        
        # 测试连接
        with GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
            driver.verify_connectivity()
            logger.info(f"✅ Neo4j连接成功: {neo4j_uri}")
            
            # 检查数据库版本
            with driver.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions")
                for record in result:
                    logger.info(f"   {record['name']}: {record['versions']}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Neo4j连接失败: {e}")
        logger.error("请确保:")
        logger.error("1. Neo4j数据库正在运行")
        logger.error("2. 连接参数正确")
        logger.error("3. 网络连通性正常")
        return False


async def create_graphrag_with_neo4j():
    """创建使用Neo4j存储的GraphRAG实例"""
    
    # 测试Neo4j连接
    if not test_neo4j_connection():
        logger.error("无法连接到Neo4j数据库，请检查配置")
        return None
    
    # 获取配置
    config = get_neo4j_config()
    
    try:
        # 创建GraphRAG实例，指定使用Neo4j图存储
        rag = GraphRAG(
            working_dir=config["working_dir"],
            enable_llm_cache=config["enable_llm_cache"],
            graph_storage_cls=Neo4jStorage,  # 指定使用Neo4j存储
            **{k: v for k, v in config.items() if k in ["addon_params", "best_model_max_token_size", "cheap_model_max_token_size", "embedding_batch_num"]}
        )
        
        logger.info("✅ GraphRAG实例创建成功，使用Neo4j图存储")
        return rag
        
    except Exception as e:
        logger.error(f"❌ GraphRAG实例创建失败: {e}")
        return None


async def demo_workflow():
    """演示完整的工作流程"""
    
    logger.info("🚀 开始Neo4j GraphRAG演示...")
    
    # 1. 创建GraphRAG实例
    rag = await create_graphrag_with_neo4j()
    if not rag:
        return
    
    # 2. 准备示例数据
    sample_text = """
    Neo4j是一个高性能的NoSQL图数据库。它使用图结构来存储数据，
    包括节点、关系和属性。Neo4j特别适合处理复杂的关联数据。
    
    GraphRAG是一种结合了知识图谱和大型语言模型的检索增强生成技术。
    它通过实体提取构建知识图谱，然后利用图结构增强检索和生成能力。
    
    在nano_graphrag项目中，我们实现了多种存储后端，包括NetworkX和Neo4j。
    Neo4j存储提供了更好的性能和可扩展性，特别适合大规模知识图谱应用。
    """
    
    # 3. 插入文档并构建知识图谱
    logger.info("📚 插入文档并构建知识图谱...")
    try:
        await rag.ainsert(sample_text)
        logger.info("✅ 知识图谱构建完成")
    except Exception as e:
        logger.error(f"❌ 知识图谱构建失败: {e}")
        return
    
    # 4. 执行不同模式的查询
    queries = [
        ("什么是Neo4j？", "naive"),
        ("GraphRAG如何工作？", "local"),
        ("Neo4j和GraphRAG之间有什么关系？", "global"),
    ]
    
    logger.info("🔍 执行不同模式的查询...")
    
    for query, mode in queries:
        try:
            logger.info(f"\n查询: {query}")
            logger.info(f"模式: {mode}")
            
            param = QueryParam(mode=mode)
            result = await rag.aquery(query, param=param)
            
            logger.info(f"结果: {result[:200]}..." if len(result) > 200 else f"结果: {result}")
            
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
    
    # 5. 查看Neo4j中的数据
    logger.info("\n📊 查看Neo4j中的知识图谱数据...")
    try:
        await inspect_neo4j_graph()
    except Exception as e:
        logger.warning(f"⚠️ 无法检查Neo4j图数据: {e}")
    
    logger.info("\n🎉 Neo4j GraphRAG演示完成!")


async def inspect_neo4j_graph():
    """检查Neo4j中的知识图谱数据"""
    try:
        from neo4j import GraphDatabase
        
        # 获取连接参数
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123456")
        
        with GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
            with driver.session() as session:
                # 统计节点数量
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                logger.info(f"   节点总数: {node_count}")
                
                # 统计关系数量
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                logger.info(f"   关系总数: {rel_count}")
                
                # 显示节点标签
                result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = [record["label"] for record in result]
                logger.info(f"   节点标签: {labels}")
                
                # 显示关系类型
                result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rel_types = [record["relationshipType"] for record in result]
                logger.info(f"   关系类型: {rel_types}")
                
                # 显示一些示例节点
                result = session.run("MATCH (n) RETURN n LIMIT 5")
                logger.info("   示例节点:")
                for i, record in enumerate(result, 1):
                    node = record["n"]
                    logger.info(f"     {i}. {dict(node)}")
                
    except Exception as e:
        logger.error(f"检查Neo4j图数据失败: {e}")


def print_setup_instructions():
    """打印设置说明"""
    print("Neo4j GraphRAG 设置说明")
    print("=" * 60)
    print()
    print("1. 启动Neo4j数据库:")
    print("   方式A - Neo4j Desktop (推荐):")  
    print("     1. 下载并安装 Neo4j Desktop")
    print("     2. 创建新的数据库实例")
    print("     3. 启动数据库")
    print("     4. 记录连接信息 (URI, 用户名, 密码)")
    print()
    print("   方式B - 使用Docker:")
    print("     cd docker")
    print("     docker-compose -f neo4j-docker-compose.yml up -d")
    print()
    print("2. 设置环境变量 (可选):")
    print("     # Neo4j Desktop")
    print("     export NEO4J_URI=neo4j://127.0.0.1:7687")
    print("     export NEO4J_USERNAME=neo4j")
    print("     export NEO4J_PASSWORD=neo4j123456")
    print()
    print("3. 验证连接:")
    print("     Neo4j Desktop: http://localhost:7474")
    print("     用户名: neo4j")
    print("     密码: neo4j123456")
    print()
    print("4. 运行示例:")
    print("     python examples/using_neo4j_graphrag.py")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo4j GraphRAG示例")
    parser.add_argument("--test-connection", action="store_true", help="只测试Neo4j连接")
    parser.add_argument("--setup-help", action="store_true", help="显示设置说明")
    
    args = parser.parse_args()
    
    if args.setup_help:
        print_setup_instructions()
    elif args.test_connection:
        test_neo4j_connection()
    else:
        # 运行完整演示
        asyncio.run(demo_workflow()) 