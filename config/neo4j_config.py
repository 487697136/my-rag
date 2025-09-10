#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j 图数据库配置文件

提供Neo4j数据库的连接配置和使用示例。
支持本地安装、Docker部署、Neo4j Aura云服务等多种部署方式。

参考文档：
- Neo4j Python Driver: https://neo4j.com/docs/getting-started/languages-guides/neo4j-python/
- Neo4j Aura: https://neo4j.com/cloud/platform/aura-graph-database/
"""

import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Neo4j数据库配置类"""
    
    # 连接配置 - Neo4j Desktop Local Instance
    uri: str = "neo4j://127.0.0.1:7687"  # Neo4j Desktop Local Instance地址
    username: str = "neo4j"              # 用户名
    password: str = "YOUR_ACTUAL_PASSWORD"  # 请替换为实际密码
    database: str = "neo4j"             # 数据库名称
    
    # 高级配置
    max_connection_pool_size: int = 50  # 连接池大小
    connection_timeout: float = 30.0    # 连接超时(秒)
    max_transaction_retry_time: float = 30.0  # 事务重试时间(秒)
    
    def to_driver_config(self) -> Tuple[str, Tuple[str, str], Dict[str, Any]]:
        """转换为Driver配置格式"""
        auth = (self.username, self.password)
        config = {
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_timeout": self.connection_timeout,
            "max_transaction_retry_time": self.max_transaction_retry_time,
        }
        return self.uri, auth, config
    
    def to_graphrag_config(self) -> Dict[str, Any]:
        """转换为GraphRAG配置格式"""
        return {
            "addon_params": {
                "neo4j_url": self.uri,
                "neo4j_auth": (self.username, self.password),
                "neo4j_database": self.database,
                "neo4j_max_connection_pool_size": self.max_connection_pool_size,
            }
        }


# 预定义配置模板
class Neo4jConfigTemplates:
    """Neo4j配置模板"""
    
    @staticmethod
    def local_development() -> Neo4jConfig:
        """本地开发环境配置（Neo4j Desktop Local Instance）"""
        return Neo4jConfig(
            uri="neo4j://127.0.0.1:7687",  # 连接到local instance
            username="neo4j",
            password="YOUR_ACTUAL_PASSWORD",  # 请替换为实际密码
            database="neo4j"  # 默认数据库，连接到instance后可访问所有数据库
        )
    
    @staticmethod
    def docker_compose() -> Neo4jConfig:
        """Docker Compose环境配置"""
        return Neo4jConfig(
            uri="bolt://localhost:7687",  # Docker使用bolt协议
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j123456"),  # 与Desktop保持一致
            database="neo4j"
        )
    
    @staticmethod
    def neo4j_aura() -> Neo4jConfig:
        """Neo4j Aura云服务配置"""
        return Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "neo4j+s://47aa7a59.databases.neo4j.io"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "1COh0KFBOJL12ByoSj82D0eoeXpFv8bWWvocts83m9s"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
    
    @staticmethod
    def production() -> Neo4jConfig:
        """生产环境配置"""
        return Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://production-server:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=100,
            connection_timeout=60.0,
            max_transaction_retry_time=60.0
        )


def get_neo4j_config(config_type: str = "aura") -> Neo4jConfig:
    """获取Neo4j配置
    
    Args:
        config_type: 配置类型 ("local", "docker", "aura", "production")
        
    Returns:
        Neo4jConfig实例
    """
    config_map = {
        "local": Neo4jConfigTemplates.local_development,
        "docker": Neo4jConfigTemplates.docker_compose,
        "aura": Neo4jConfigTemplates.neo4j_aura,
        "production": Neo4jConfigTemplates.production,
    }
    
    if config_type not in config_map:
        raise ValueError(f"不支持的配置类型: {config_type}")
    
    return config_map[config_type]()


def test_neo4j_connection(config: Neo4jConfig) -> bool:
    """测试Neo4j连接
    
    Args:
        config: Neo4j配置
        
    Returns:
        连接是否成功
    """
    try:
        from neo4j import GraphDatabase
        
        uri, auth, driver_config = config.to_driver_config()
        
        with GraphDatabase.driver(uri, auth=auth, **driver_config) as driver:
            driver.verify_connectivity()
            print(f"✅ Neo4j连接成功: {uri}")
            return True
            
    except Exception as e:
        print(f"❌ Neo4j连接失败: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    print("Neo4j配置测试")
    print("=" * 50)
    
    # 测试Aura配置
    print("\n测试Neo4j Aura云端配置...")
    aura_config = get_neo4j_config("aura")
    print(f"URI: {aura_config.uri}")
    print(f"Username: {aura_config.username}")
    print(f"Database: {aura_config.database}")
    
    # 如果Neo4j正在运行，可以取消注释下面的行来测试连接
    # test_neo4j_connection(aura_config)
    
    # 显示GraphRAG配置格式
    print("\nGraphRAG配置格式:")
    graphrag_config = aura_config.to_graphrag_config()
    import json
    print(json.dumps(graphrag_config, indent=2, ensure_ascii=False)) 