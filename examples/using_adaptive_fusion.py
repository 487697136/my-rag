"""
示例：使用自适应融合策略

本示例展示了如何使用动态权重分配和自适应融合策略进行检索结果融合。
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_graphrag.complexity.router import ComplexityAwareRouter
from nano_graphrag.hybrid.multi_retrieval import ParallelRetriever, RetrievalResult
from nano_graphrag.hybrid.fusion import LinearFusion, RRFFusion
from nano_graphrag.hybrid.adaptive_weight import AdaptiveWeightManager, FeedbackDrivenFusion
from nano_graphrag.hybrid.dynamic_fusion import DynamicWeightFusion, ConfidenceAwareFusion, AdaptiveFusionStrategy
from nano_graphrag.evaluation.metrics import RetrievalMetrics
from nano_graphrag.feedback.processor import FeedbackProcessor
from nano_graphrag.feedback.collector import FeedbackCollector
from nano_graphrag.feedback.storage import FeedbackStorage

# 创建模拟检索器结果
def create_mock_retrieval_result(query: str, retriever_id: str, doc_count: int = 10, relevance_pattern: str = 'random') -> RetrievalResult:
    """创建模拟检索结果"""
    documents = []
    
    for i in range(doc_count):
        # 生成文档ID
        doc_id = f"{retriever_id}_doc_{i}"
        
        # 根据模式生成相关性分数
        if relevance_pattern == 'descending':
            score = 1.0 - (i / doc_count)
        elif relevance_pattern == 'ascending':
            score = i / doc_count
        elif relevance_pattern == 'middle_peak':
            # 中间位置的文档有更高的相关性
            mid = doc_count / 2
            score = 1.0 - abs(i - mid) / mid
        else:  # random
            score = np.random.random()
        
        # 创建文档
        document = {
            'id': doc_id,
            'content': f"模拟文档内容 {i} 来自 {retriever_id}，查询：{query}",
            'score': score,
            'metadata': {
                'retriever': retriever_id,
                'rank': i,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        documents.append(document)
    
    metrics = {
        'execution_time': np.random.random() * 0.5,  # 模拟执行时间
        'doc_count': doc_count
    }
    
    return RetrievalResult(documents, retriever_id, query, metrics)

# 模拟检索器函数
async def mock_retriever(query: str, retriever_id: str, delay: float = 0.1, doc_count: int = 10, relevance_pattern: str = 'random') -> RetrievalResult:
    """模拟检索器函数"""
    # 模拟延迟
    await asyncio.sleep(delay)
    
    # 生成检索结果
    return create_mock_retrieval_result(query, retriever_id, doc_count, relevance_pattern)

# 模拟复杂度分类器
class MockComplexityClassifier:
    """模拟复杂度分类器"""
    
    def __init__(self):
        """初始化模拟复杂度分类器"""
        self.patterns = {
            'what': 'zero_hop',
            'who': 'zero_hop',
            'when': 'zero_hop',
            'where': 'one_hop',
            'which': 'one_hop',
            'how': 'one_hop',
            'why': 'multi_hop',
            'relationship': 'multi_hop',
            'compare': 'multi_hop'
        }
    
    async def predict_complexity(self, query: str) -> Dict[str, Any]:
        """预测查询复杂度"""
        # 根据关键词判断复杂度
        query_lower = query.lower()
        complexity = 'one_hop'  # 默认复杂度
        confidence = 0.7  # 默认置信度
        
        for keyword, complexity_class in self.patterns.items():
            if keyword in query_lower:
                complexity = complexity_class
                confidence = 0.8 + 0.1 * np.random.random()  # 0.8-0.9之间的随机数
                break
        
        # 长查询通常更复杂
        if len(query.split()) > 10:
            if complexity != 'multi_hop':
                complexity = 'multi_hop'
                confidence -= 0.1
        
        # 确保置信度在0-1之间
        confidence = max(0.4, min(0.95, confidence))
        
        return {
            'complexity': complexity,
            'confidence': confidence,
            'query': query
        }

# 模拟用户反馈函数
def generate_mock_feedback(query: str, retrieval_results: Dict[str, RetrievalResult], fused_results: List[Dict[str, Any]], complexity: str) -> Dict[str, Any]:
    """生成模拟用户反馈"""
    # 选择一些点击的文档
    num_clicks = np.random.randint(0, 3)  # 0-2个点击
    clicked_docs = []
    
    if fused_results and num_clicks > 0:
        # 从前5个结果中随机选择
        top_n = min(5, len(fused_results))
        indices = np.random.choice(top_n, size=min(num_clicks, top_n), replace=False)
        
        for idx in indices:
            if idx < len(fused_results):
                doc = fused_results[idx]
                clicked_docs.append(doc.get('id', ''))
    
    # 生成评分 (1-5)
    # 复杂度越高，评分倾向于越低（模拟复杂查询的不满意度更高）
    if complexity == 'zero_hop':
        rating_base = 4.0
    elif complexity == 'one_hop':
        rating_base = 3.5
    else:
        rating_base = 3.0
        
    # 根据点击数调整评分
    if num_clicks > 0:
        rating_base += 0.5
    
    # 添加随机波动
    rating = max(1.0, min(5.0, rating_base + (np.random.random() - 0.5)))
    
    # 构建反馈
    feedback = {
        'user_rating': rating,
        'clicks': clicked_docs,
        'dwell_times': {doc_id: 5.0 + 10.0 * np.random.random() for doc_id in clicked_docs},  # 5-15秒的停留时间
        'complexity': complexity,
        'timestamp': datetime.now().isoformat()
    }
    
    return feedback

# 主示例函数
async def main():
    """主示例函数"""
    print("初始化组件...")
    
    # 创建并行检索器
    retriever_ids = ['naive', 'bm25', 'local', 'global', 'llm_only']
    parallel_retriever = ParallelRetriever({}, max_concurrency=5)
    
    # 注册检索器
    for retriever_id in retriever_ids:
        parallel_retriever.add_retriever(
            retriever_id,
            lambda q, rid=retriever_id: mock_retriever(
                q, 
                rid, 
                delay=0.1 + 0.2 * np.random.random(),  # 0.1-0.3秒的随机延迟
                doc_count=np.random.randint(5, 15)  # 5-15个文档
            )
        )
    
    # 创建融合策略
    linear_fusion = LinearFusion()
    rrf_fusion = RRFFusion()
    dynamic_fusion = DynamicWeightFusion(retriever_ids)
    
    # 创建复杂度感知融合策略
    confidence_fusion = ConfidenceAwareFusion(retriever_ids)
    
    # 创建复杂度分类器
    complexity_classifier = MockComplexityClassifier()
    
    # 创建反馈存储和处理器
    feedback_storage = FeedbackStorage("feedback_data")
    feedback_processor = FeedbackProcessor(feedback_file="feedback_data/feedback.json")
    
    # 创建反馈收集器
    feedback_collector = FeedbackCollector(storage=feedback_storage)
    
    # 创建权重管理器
    weight_manager = AdaptiveWeightManager(
        retriever_ids=retriever_ids,
        optimizer_type="adaptive",
        save_path="models/weight_optimizer.json",
        feedback_processor=feedback_processor
    )
    
    # 创建自适应融合策略
    adaptive_fusion = AdaptiveFusionStrategy(
        weight_manager=weight_manager,
        feedback_processor=feedback_processor,
        linear_fusion=linear_fusion,
        rrf_fusion=rrf_fusion
    )
    
    # 创建评估指标
    metrics = RetrievalMetrics()
    
    # 运行一系列模拟查询
    test_queries = [
        "什么是纳米材料",  # zero_hop
        "谁发明了相对论",  # zero_hop
        "太阳系中最大的行星是哪个",  # zero_hop
        "中国和美国的经济关系如何",  # one_hop
        "量子计算机如何工作",  # one_hop
        "为什么天空是蓝色的",  # multi_hop
        "比较传统燃料车和电动车对环境的影响",  # multi_hop
        "区块链技术对金融行业的长期影响是什么"  # multi_hop
    ]
    
    print("开始进行模拟查询测试...")
    results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试查询 {i}/{len(test_queries)}: '{query}'")
        
        # 预测复杂度
        prediction = await complexity_classifier.predict_complexity(query)
        complexity = prediction['complexity']
        confidence = prediction['confidence']
        
        print(f"预测复杂度: {complexity}, 置信度: {confidence:.2f}")
        
        # 并行检索
        print("执行并行检索...")
        retrieval_results = await parallel_retriever.retrieve_all(query)
        
        print("检索结果统计:")
        for retriever_id, result in retrieval_results.items():
            print(f"  - {retriever_id}: {result.count} 个文档, 耗时 {result.metrics.get('execution_time', 0):.2f}s")
        
        # 使用不同融合策略
        print("应用不同融合策略...")
        
        # 使用线性融合
        linear_result = await linear_fusion.fuse(retrieval_results, query)
        
        # 使用RRF融合
        rrf_result = await rrf_fusion.fuse(retrieval_results, query)
        
        # 使用动态权重融合
        dynamic_result = await dynamic_fusion.fuse(retrieval_results, query, {'complexity': complexity, 'confidence': confidence})
        
        # 使用置信度感知融合
        confidence_result = await confidence_fusion.fuse(retrieval_results, query, {'complexity': complexity, 'confidence': confidence})
        
        # 使用自适应融合
        adaptive_result = await adaptive_fusion.fuse(retrieval_results, query, {'complexity': complexity, 'confidence': confidence})
        
        # 比较融合结果
        fusion_results = {
            'linear': linear_result,
            'rrf': rrf_result,
            'dynamic': dynamic_result,
            'confidence': confidence_result,
            'adaptive': adaptive_result
        }
        
        print("融合结果比较:")
        for fusion_name, fusion_result in fusion_results.items():
            print(f"  - {fusion_name}: {fusion_result.count} 个文档")
            
            # 查看Top-3文档来源
            if fusion_result.count > 0:
                top_docs = fusion_result.documents[:3]
                sources = [doc.get('sources', [doc.get('retriever', 'unknown')]) for doc in top_docs]
                print(f"    Top-3文档来源: {sources}")
        
        # 生成模拟反馈
        feedback = generate_mock_feedback(query, retrieval_results, adaptive_result.documents, complexity)
        print(f"模拟用户反馈: 评分={feedback['user_rating']:.1f}, 点击={len(feedback['clicks'])}个文档")
        
        # 处理反馈
        adaptive_fusion.process_feedback(query, adaptive_result, feedback)
        
        # 存储结果
        results[query] = {
            'complexity': complexity,
            'confidence': confidence,
            'feedback': feedback,
            'fusion_results': {name: result.metrics for name, result in fusion_results.items()}
        }
    
    # 获取使用统计
    usage_stats = adaptive_fusion.get_usage_stats()
    
    print("\n使用统计:")
    print(f"总查询次数: {usage_stats.get('total_queries', 0)}")
    print(f"策略使用频率: {usage_stats.get('strategy_frequency', {})}")
    print(f"复杂度分布: {usage_stats.get('complexity_distribution', {})}")
    print(f"平均置信度: {usage_stats.get('average_confidence', 0):.2f}")
    
    # 获取权重管理器性能统计
    weight_stats = weight_manager.get_performance_stats()
    
    print("\n权重管理器性能统计:")
    print(f"查询计数: {weight_stats.get('query_count', 0)}")
    print(f"复杂度分布: {weight_stats.get('recent_complexity_distribution', {})}")
    
    # 保存结果
    with open('results/adaptive_fusion_results.json', 'w', encoding='utf-8') as f:
        # 转换非序列化对象
        serialized_results = {}
        for query, result in results.items():
            serialized_results[query] = {
                'complexity': result['complexity'],
                'confidence': result['confidence'],
                'feedback': result['feedback'],
                'fusion_results': {name: {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, list, dict))}
                                 for name, metrics in result['fusion_results'].items()}
            }
        
        json.dump(serialized_results, f, indent=2, ensure_ascii=False)
    
    print("\n结果已保存到 results/adaptive_fusion_results.json")

# 可视化功能
def visualize_results(results_file: str = 'results/adaptive_fusion_results.json'):
    """可视化结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取评分数据
    queries = list(results.keys())
    complexities = [results[q]['complexity'] for q in queries]
    ratings = [results[q]['feedback']['user_rating'] for q in queries]
    
    # 创建复杂度颜色映射
    color_map = {
        'zero_hop': 'green',
        'one_hop': 'blue',
        'multi_hop': 'red'
    }
    colors = [color_map[c] for c in complexities]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(queries)), ratings, color=colors)
    plt.xlabel('查询')
    plt.ylabel('用户评分')
    plt.title('不同复杂度查询的用户评分')
    plt.xticks(range(len(queries)), [q[:10] + '...' for q in queries], rotation=45)
    plt.tight_layout()
    
    # 添加复杂度图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['zero_hop'], label='Zero-hop'),
        Patch(facecolor=color_map['one_hop'], label='One-hop'),
        Patch(facecolor=color_map['multi_hop'], label='Multi-hop')
    ]
    plt.legend(handles=legend_elements)
    
    # 保存图表
    plt.savefig('results/adaptive_fusion_ratings.png')
    print("图表已保存到 results/adaptive_fusion_ratings.png")

# 运行示例
if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    os.makedirs('feedback_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 运行主函数
    asyncio.run(main())
    
    # 可视化结果
    try:
        visualize_results()
    except Exception as e:
        print(f"可视化失败: {e}") 