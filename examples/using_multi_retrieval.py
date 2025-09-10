#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用多源检索融合功能示例
"""

import os
import asyncio
import logging
from typing import Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在路径中
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入nano-GraphRAG
from nano_graphrag.graphrag import GraphRAG
from nano_graphrag.hybrid.graphrag_hybrid import GraphRAGHybrid, HybridQueryParam
from nano_graphrag.complexity.router import ComplexityAwareRouter
from nano_graphrag.hybrid.adaptive_router import ConfidenceAwareRouter


async def demo_single_mode_query(rag: GraphRAG):
    """演示单一检索模式查询"""
    print("\n=== 单一检索模式查询 ===")
    
    # 不同类型的查询
    queries = [
        ("什么是计算机科学?", "zero_hop"),  # 简单查询，直接LLM回答
        ("生成式AI的发展历程是什么?", "one_hop"),  # 单跳查询，向量检索
        ("量子计算与经典计算的主要区别是什么?", "multi_hop")  # 多跳查询，图检索
    ]
    
    for query, expected_complexity in queries:
        print(f"\n查询: {query}")
        print(f"预期复杂度: {expected_complexity}")
        
        # 创建查询参数
        param = rag.create_query_param(query)
        print(f"选择检索模式: {param.mode}")
        
        # 执行查询
        response = await rag.aquery(query, param)
        print(f"回答: {response[:100]}...")


async def demo_hybrid_query(hybrid_rag: GraphRAGHybrid):
    """演示混合检索查询"""
    print("\n=== 混合检索查询 ===")
    
    # 不同类型的查询
    queries = [
        "人工智能的伦理问题有哪些?",
        "解释量子纠缠现象",
        "自然语言处理的关键技术有哪些?"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        
        # 创建混合查询参数
        param = hybrid_rag.create_hybrid_query_param(query)
        
        # 执行混合检索查询
        response = await hybrid_rag.hybrid_query(query, param)
        print(f"回答: {response[:100]}...")


async def demo_confidence_routing(hybrid_rag: GraphRAGHybrid):
    """演示置信度路由和动态权重分配"""
    print("\n=== 置信度路由和动态权重分配 ===")
    
    # 不同类型的查询
    queries = [
        "什么是机器学习?",
        "深度学习与传统机器学习的区别?",
        "解释卷积神经网络是如何进行图像识别的"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        
        # 获取置信度路由结果
        complexity_result = await hybrid_rag.confidence_router.predict_complexity_with_calibration(query)
        
        print(f"复杂度: {complexity_result['complexity']}")
        print(f"原始置信度: {complexity_result['original_confidence']:.4f}")
        print(f"校准置信度: {complexity_result['confidence']:.4f}")
        
        # 获取动态权重
        selected_mode, weights, confidence = await hybrid_rag.confidence_router.route_with_weights(query)
        
        print(f"选择检索模式: {selected_mode}")
        print(f"动态权重分配: {weights}")
        
        # 创建使用动态权重的查询参数
        param = HybridQueryParam(
            fusion_method="linear",
            weights=weights
        )
        
        # 执行混合检索查询
        response = await hybrid_rag.hybrid_query(query, param)
        print(f"回答: {response[:100]}...")


async def demo_different_fusion_methods(hybrid_rag: GraphRAGHybrid):
    """演示不同的融合策略"""
    print("\n=== 不同的融合策略 ===")
    
    query = "比较传统数据库和图数据库的优缺点"
    print(f"查询: {query}")
    
    # 线性融合
    print("\n1. 线性融合:")
    param_linear = HybridQueryParam(
        fusion_method="linear",
        weights={
            "naive": 1.0,
            "bm25": 0.8,
            "local": 0.6,
            "global": 0.7
        }
    )
    response_linear = await hybrid_rag.hybrid_query(query, param_linear)
    print(f"回答: {response_linear[:100]}...")
    
    # RRF融合
    print("\n2. RRF融合:")
    param_rrf = HybridQueryParam(
        fusion_method="rrf",
        rrf_k=60.0,
        weights={
            "naive": 1.0,
            "bm25": 0.8,
            "local": 0.6,
            "global": 0.7
        }
    )
    response_rrf = await hybrid_rag.hybrid_query(query, param_rrf)
    print(f"回答: {response_rrf[:100]}...")
    
    # 动态权重融合
    print("\n3. 动态权重融合:")
    param_dynamic = HybridQueryParam(
        fusion_method="dynamic",
        dynamic_weight_gamma=2.0
    )
    response_dynamic = await hybrid_rag.hybrid_query(query, param_dynamic)
    print(f"回答: {response_dynamic[:100]}...")


async def main():
    """主函数"""
    print("初始化nano-GraphRAG...")
    
    # 数据目录
    cache_dir = "./data_cache/novel_rag_benchmark_cache"
    
    # 创建常规GraphRAG
    rag = GraphRAG(
        working_dir=cache_dir,
        router_cls=ComplexityAwareRouter,
        router_kwargs={
            "model_path": "nano_graphrag/models/modernbert_complexity_classifier",
            "confidence_threshold": 0.6,
            "enable_fallback": True,
            "use_modernbert": True
        }
    )
    
    # 创建混合检索GraphRAG
    hybrid_rag = GraphRAGHybrid(
        working_dir=cache_dir,
        router_cls=ConfidenceAwareRouter,
        router_kwargs={
            "model_path": "nano_graphrag/models/modernbert_complexity_classifier",
            "confidence_threshold": 0.6,
            "enable_fallback": True,
            "use_modernbert": True,
            "calibration_path": "stage1/results/calibration_results.json"
        }
    )
    
    # 运行演示
    await demo_single_mode_query(rag)
    await demo_hybrid_query(hybrid_rag)
    await demo_confidence_routing(hybrid_rag)
    await demo_different_fusion_methods(hybrid_rag)


if __name__ == "__main__":
    asyncio.run(main()) 