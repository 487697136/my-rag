#!/usr/bin/env python3
"""
Nano GraphRAG - 快速开始示例

这个示例展示了如何使用增强版的GraphRAG系统：
1. 基础使用方法
2. 增强功能的使用
3. 评估和优化

适合本科生学习和研究使用。
"""

import asyncio
import os
from typing import List, Dict, Any

# 导入Nano GraphRAG
from nano_graphrag import (
    create_nano_graphrag,
    ModernEvaluator,
    create_modern_evaluator,
    QueryParam
)


async def basic_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 1. 创建GraphRAG实例（增强版）
    rag = create_nano_graphrag(
        working_dir="./nano_graphrag_data",
        enable_enhanced_features=True,  # 启用增强功能
        enable_naive_rag=True,          # 启用多种检索方式
        enable_local=True
    )
    
    # 2. 准备示例文档
    documents = [
        """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        致力于创建能够执行通常需要人类智能的任务的机器和程序。
        AI的主要领域包括机器学习、深度学习、自然语言处理、计算机视觉等。
        """,
        """
        机器学习是人工智能的一个子领域，它使用算法和统计模型来分析和学习数据，
        从而在没有明确编程的情况下提高任务性能。常见的机器学习方法包括监督学习、
        无监督学习和强化学习。
        """,
        """
        深度学习是机器学习的一个子集，使用多层神经网络来学习数据的表示。
        深度学习在图像识别、语音识别、自然语言处理等领域取得了重大突破。
        著名的深度学习模型包括卷积神经网络（CNN）和循环神经网络（RNN）。
        """,
        """
        自然语言处理（NLP）是人工智能的一个分支，专注于使计算机能够理解、
        解释和生成人类语言。NLP的应用包括机器翻译、情感分析、文本摘要、
        问答系统等。现代NLP大量使用Transformer模型，如BERT、GPT等。
        """
    ]
    
    # 3. 插入文档到知识库
    print("插入文档到知识库...")
    await rag.ainsert(documents)
    print("文档插入完成！")
    
    # 4. 进行查询
    queries = [
        "什么是人工智能？",                    # 简单查询
        "机器学习和深度学习的关系是什么？",      # 中等复杂度查询  
        "深度学习在NLP中的应用有哪些？"        # 复杂查询
    ]
    
    print("\n开始查询测试...")
    for i, query in enumerate(queries, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        
        # 系统会自动选择最佳的检索模式
        response = await rag.aquery(query)
        print(f"回答: {response}")
    
    return rag


async def enhanced_features_example():
    """增强功能示例"""
    print("\n\n=== 增强功能示例 ===")
    
    # 创建增强版实例
    rag = create_nano_graphrag(
        working_dir="./nano_graphrag_enhanced",
        enable_enhanced_features=True,
    )
    
    # 示例数据
    test_documents = [
        "量子计算是一种利用量子力学现象进行计算的新型计算范式。",
        "量子比特（qubit）是量子计算的基本单位，可以同时处于0和1的叠加态。",
        "量子纠缠是量子计算中的重要现象，使得量子比特之间可以保持神秘的关联。",
        "量子算法如Shor算法和Grover算法在特定问题上比经典算法具有指数级优势。"
    ]
    
    await rag.ainsert(test_documents)
    
    # 使用不同的查询参数测试
    query = "量子计算的基本原理是什么？"
    
    # 测试不同的检索模式
    modes = ["naive", "local", "global"]
    results = {}
    
    for mode in modes:
        try:
            param = QueryParam(mode=mode)
            response = await rag.aquery(query, param)
            results[mode] = response
            print(f"\n{mode.upper()} 模式回答:")
            print(f"{response}")
        except Exception as e:
            print(f"{mode} 模式出错: {e}")
    
    # 获取系统统计信息
    stats = rag.get_system_statistics()
    print(f"\n系统统计信息:")
    print(f"- 工作目录: {stats['basic_info']['working_dir']}")
    print(f"- 增强功能状态: {stats['basic_info']['enable_enhanced_features']}")
    
    return rag, results


async def evaluation_example():
    """评估示例"""
    print("\n\n=== 评估示例 ===")
    
    # 创建现代评估器
    evaluator = create_modern_evaluator()
    
    # 模拟评估数据
    questions = [
        "什么是机器学习？",
        "深度学习和传统机器学习的区别？",
        "自然语言处理的主要应用？"
    ]
    
    answers = [
        "机器学习是人工智能的一个分支，通过算法从数据中学习模式。",
        "深度学习使用多层神经网络，比传统机器学习在某些任务上表现更好。",
        "自然语言处理应用包括机器翻译、情感分析、文本摘要等。"
    ]
    
    contexts_list = [
        ["机器学习是AI的子领域", "使用算法和统计模型学习数据"],
        ["深度学习使用神经网络", "传统ML使用统计方法", "深度学习在图像识别效果好"],
        ["NLP处理人类语言", "应用于翻译、摘要", "使用Transformer模型"]
    ]
    
    complexity_types = ["one_hop", "multi_hop", "one_hop"]
    
    # 批量评估
    evaluation_results = evaluator.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        complexity_types=complexity_types
    )
    
    # 显示评估结果
    print("批量评估结果:")
    batch_stats = evaluation_results['batch_statistics']
    
    for metric, stats in batch_stats.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"- {metric}: 平均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
    
    # 复杂度分解统计
    if 'complexity_breakdown' in batch_stats:
        print("\n按复杂度分类的结果:")
        for complexity, data in batch_stats['complexity_breakdown'].items():
            print(f"- {complexity}: 数量={data['count']}, 平均分={data['mean_score']:.3f}")
    
    return evaluation_results


async def complete_workflow_example():
    """完整工作流示例"""
    print("\n\n=== 完整工作流示例 ===")
    
    # 1. 创建系统
    rag = create_nano_graphrag(
        working_dir="./nano_graphrag_complete",
        enable_enhanced_features=True,
    )
    
    # 2. 准备更丰富的数据集
    ai_knowledge = [
        """
        人工智能的历史可以追溯到1950年代，当时阿兰·图灵提出了著名的图灵测试。
        1956年，约翰·麦卡锡在达特茅斯会议上首次提出"人工智能"这个术语。
        早期的AI研究主要集中在符号推理和专家系统上。
        """,
        """
        1980年代，机器学习开始兴起，特别是统计学习方法的发展。
        1997年，IBM的深蓝计算机击败了国际象棋世界冠军加里·卡斯帕罗夫，
        标志着AI在特定领域达到了人类水平。
        """,
        """
        2006年，杰弗里·辛顿提出了深度学习的概念，开启了深度学习时代。
        2012年，AlexNet在ImageNet竞赛中的胜利证明了深度学习在计算机视觉中的潜力。
        2016年，AlphaGo击败围棋世界冠军李世石，展示了AI在复杂策略游戏中的能力。
        """,
        """
        2017年，Google提出了Transformer架构，彻底改变了自然语言处理领域。
        2018年，BERT模型的发布进一步推动了NLP技术的发展。
        2020年，GPT-3的发布展示了大型语言模型的强大能力。
        """,
        """
        现代AI应用广泛，包括计算机视觉、自然语言处理、语音识别、推荐系统、
        自动驾驶、医疗诊断、金融分析等领域。AI技术正在深刻改变我们的生活和工作方式。
        """
    ]
    
    print("插入AI知识库...")
    await rag.ainsert(ai_knowledge)
    
    # 3. 设计测试查询（不同复杂度）
    test_queries = [
        ("谁是人工智能之父？", "zero_hop"),                # 简单事实查询
        ("深度学习的发展历程是什么？", "one_hop"),          # 单步推理
        ("AI技术发展对现代社会有什么影响？", "multi_hop"),   # 多步推理
    ]
    
    # 4. 执行查询并收集结果
    query_results = []
    for query, expected_complexity in test_queries:
        print(f"\n查询: {query}")
        response = await rag.aquery(query)
        print(f"回答: {response}")
        
        query_results.append({
            'query': query,
            'response': response,
            'expected_complexity': expected_complexity
        })
    
    # 5. 如果有评估功能，进行系统评估
    if rag.enable_enhanced_features and hasattr(rag, 'modern_evaluator'):
        print("\n进行系统评估...")
        
        # 模拟标准答案用于评估
        evaluation_data = {
            'questions': [result['query'] for result in query_results],
            'answers': [result['response'] for result in query_results],
            'contexts_list': [["相关上下文1", "相关上下文2"] for _ in query_results],
            'complexity_types': [result['expected_complexity'] for result in query_results]
        }
        
        try:
            eval_results = await rag.evaluate_system(**evaluation_data)
            if 'error' not in eval_results:
                print("评估完成！")
                print(f"总评估数量: {eval_results.get('total_evaluated', 0)}")
            else:
                print(f"评估失败: {eval_results['error']}")
        except Exception as e:
            print(f"评估过程出错: {e}")
    
    # 6. 显示系统统计
    stats = rag.get_system_statistics()
    print(f"\n=== 系统统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return rag, query_results


def main():
    """主函数"""
    print("🚀 Nano GraphRAG - 增强版快速开始示例")
    print("=" * 50)
    
    # 设置环境（如果需要）
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    async def run_all_examples():
        try:
            # 运行基础示例
            await basic_example()
            
            # 运行增强功能示例
            await enhanced_features_example()
            
            # 运行评估示例
            await evaluation_example()
            
            # 运行完整工作流示例
            await complete_workflow_example()
            
            print("\n✅ 所有示例运行完成！")
            print("\n📚 接下来你可以：")
            print("1. 修改文档内容，测试不同的知识领域")
            print("2. 尝试不同的查询复杂度")
            print("3. 调整系统配置参数")
            print("4. 集成到你的项目中")
            
        except Exception as e:
            print(f"❌ 示例运行出错: {e}")
            print("请检查：")
            print("1. 是否安装了所有依赖？")
            print("2. 是否设置了API密钥（如果使用外部LLM）？")
            print("3. 是否有足够的存储空间？")
    
    # 运行异步示例
    asyncio.run(run_all_examples())


if __name__ == "__main__":
    main() 