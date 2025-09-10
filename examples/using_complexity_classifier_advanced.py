"""
高级复杂度分类器RAG流程示例

本示例提供更高级的功能：
1. 支持从多种格式文件加载文档（txt, json, jsonl）
2. 自定义复杂度分类器配置
3. 详细的性能分析和可视化
4. 批量查询测试和结果对比
5. 系统状态监控和调试

使用方法：
python examples/using_complexity_classifier_advanced.py --docs_path ./your_documents/ --output_dir ./results/
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.complexity import ComplexityAwareRouter, ComplexityClassifier
from nano_graphrag._utils import logger

# 配置日志
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


#------------------------------------------------------------------------------
# 文档加载器
#------------------------------------------------------------------------------

class DocumentLoader:
    """文档加载器，支持多种格式"""
    
    @staticmethod
    def load_from_txt(file_path: str) -> List[str]:
        """从txt文件加载文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 按段落分割
            documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        return documents
    
    @staticmethod
    def load_from_json(file_path: str) -> List[str]:
        """从JSON文件加载文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 提取文本内容
                    if 'content' in item:
                        documents.append(item['content'])
                    elif 'text' in item:
                        documents.append(item['text'])
                    elif 'title' in item and 'content' in item:
                        documents.append(f"标题：{item['title']}\n\n内容：{item['content']}")
                elif isinstance(item, str):
                    documents.append(item)
        elif isinstance(data, dict):
            # 单个文档
            if 'content' in data:
                documents.append(data['content'])
            elif 'text' in data:
                documents.append(data['text'])
        
        return documents
    
    @staticmethod
    def load_from_jsonl(file_path: str) -> List[str]:
        """从JSONL文件加载文档"""
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            if 'content' in item:
                                documents.append(item['content'])
                            elif 'text' in item:
                                documents.append(item['text'])
                        elif isinstance(item, str):
                            documents.append(item)
                    except json.JSONDecodeError:
                        continue
        return documents
    
    @staticmethod
    def load_documents(docs_path: str) -> List[str]:
        """从路径加载文档，自动检测格式"""
        docs_path = Path(docs_path)
        documents = []
        
        if docs_path.is_file():
            # 单个文件
            suffix = docs_path.suffix.lower()
            if suffix == '.txt':
                documents = DocumentLoader.load_from_txt(str(docs_path))
            elif suffix == '.json':
                documents = DocumentLoader.load_from_json(str(docs_path))
            elif suffix == '.jsonl':
                documents = DocumentLoader.load_from_jsonl(str(docs_path))
        elif docs_path.is_dir():
            # 目录中的所有文件
            for file_path in docs_path.rglob('*'):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    try:
                        if suffix == '.txt':
                            docs = DocumentLoader.load_from_txt(str(file_path))
                        elif suffix == '.json':
                            docs = DocumentLoader.load_from_json(str(file_path))
                        elif suffix == '.jsonl':
                            docs = DocumentLoader.load_from_jsonl(str(file_path))
                        else:
                            continue
                        documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"加载文件 {file_path} 失败: {e}")
        
        return documents


#------------------------------------------------------------------------------
# 复杂度分类器配置
#------------------------------------------------------------------------------

class ComplexityClassifierConfig:
    """复杂度分类器配置类"""
    
    def __init__(self, 
                 model_path: str = "nano_graphrag/models/modernbert_complexity_classifier",
                 confidence_threshold: float = 0.6,
                 enable_fallback: bool = True,
                 use_modernbert: bool = True):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.use_modernbert = use_modernbert
    
    def create_router(self) -> ComplexityAwareRouter:
        """创建复杂度感知路由器"""
        return ComplexityAwareRouter(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            enable_fallback=self.enable_fallback,
            use_modernbert=self.use_modernbert
        )


#------------------------------------------------------------------------------
# RAG系统构建器
#------------------------------------------------------------------------------

class RAGSystemBuilder:
    """RAG系统构建器"""
    
    def __init__(self, working_dir: str, config: ComplexityClassifierConfig):
        self.working_dir = working_dir
        self.config = config
        self.rag = None
    
    def build(self) -> GraphRAG:
        """构建RAG系统"""
        print(f"🔧 构建RAG系统，工作目录: {self.working_dir}")
        
        # 创建GraphRAG实例
        self.rag = GraphRAG(
            working_dir=self.working_dir,
            enable_naive_rag=True,
            enable_bm25=True,
            enable_local=True,
            router_cls=ComplexityAwareRouter,
            router_kwargs={
                "model_path": self.config.model_path,
                "confidence_threshold": self.config.confidence_threshold,
                "enable_fallback": self.config.enable_fallback,
                "use_modernbert": self.config.use_modernbert
            }
        )
        
        return self.rag
    
    def insert_documents(self, documents: List[str]) -> None:
        """插入文档"""
        if not self.rag:
            raise ValueError("RAG系统未构建，请先调用build()")
        
        print(f"📚 插入 {len(documents)} 个文档")
        
        start_time = time.time()
        for i, doc in enumerate(documents):
            print(f"正在处理文档 {i+1}/{len(documents)}...")
            self.rag.insert(doc)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 文档插入完成，耗时: {elapsed_time:.2f}秒")


#------------------------------------------------------------------------------
# 查询测试器
#------------------------------------------------------------------------------

class QueryTester:
    """查询测试器"""
    
    def __init__(self, rag: GraphRAG):
        self.rag = rag
        self.results = []
    
    async def test_complexity_classification(self, queries: List[str]) -> Dict[str, Any]:
        """测试复杂度分类"""
        print("\n🧠 测试复杂度分类器")
        print("="*60)
        
        results = []
        for query in queries:
            complexity_result = await self.rag.router.predict_complexity_detailed(query)
            results.append({
                'query': query,
                'complexity': complexity_result['complexity'],
                'confidence': complexity_result['confidence'],
                'candidate_modes': complexity_result['candidate_modes'],
                'method': complexity_result['method']
            })
            
            print(f"\n查询: {query}")
            print(f"预测复杂度: {complexity_result['complexity']}")
            print(f"置信度: {complexity_result['confidence']:.3f}")
            print(f"候选模式: {complexity_result['candidate_modes']}")
        
        return results
    
    async def test_queries_with_routing(self, queries: List[str]) -> List[Dict[str, Any]]:
        """使用复杂度路由测试查询"""
        print("\n🔍 测试复杂度感知查询路由")
        print("="*60)
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")
            
            start_time = time.time()
            try:
                response = await self.rag.aquery(query)
                elapsed_time = time.time() - start_time
                
                result = {
                    'query': query,
                    'response': response,
                    'elapsed_time': elapsed_time,
                    'success': True
                }
                results.append(result)
                
                print(f"回答: {response}")
                print(f"耗时: {elapsed_time:.2f}秒")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result = {
                    'query': query,
                    'response': str(e),
                    'elapsed_time': elapsed_time,
                    'success': False
                }
                results.append(result)
                
                print(f"查询失败: {e}")
        
        return results
    
    def test_different_modes(self, query: str, modes: List[str] = None) -> List[Dict[str, Any]]:
        """测试不同检索模式"""
        if modes is None:
            modes = ["llm_only", "naive", "bm25", "local", "global"]
        
        print(f"\n🔄 测试不同检索模式: {query}")
        print("="*60)
        
        results = []
        for mode in modes:
            print(f"\n--- 模式: {mode} ---")
            
            start_time = time.time()
            try:
                response = self.rag.query(query, param=QueryParam(mode=mode))
                elapsed_time = time.time() - start_time
                
                result = {
                    'mode': mode,
                    'response': response,
                    'elapsed_time': elapsed_time,
                    'success': True
                }
                results.append(result)
                
                print(f"回答: {response}")
                print(f"耗时: {elapsed_time:.2f}秒")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result = {
                    'mode': mode,
                    'response': str(e),
                    'elapsed_time': elapsed_time,
                    'success': False
                }
                results.append(result)
                
                print(f"查询失败: {e}")
        
        return results


#------------------------------------------------------------------------------
# 性能分析器
#------------------------------------------------------------------------------

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, rag: GraphRAG, output_dir: str):
        self.rag = rag
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """分析系统性能"""
        print("\n📊 系统性能分析")
        print("="*60)
        
        # 获取复杂度统计
        complexity_stats = {}
        if hasattr(self.rag.router, 'get_complexity_stats'):
            complexity_stats = self.rag.router.get_complexity_stats()
            print(f"复杂度分类统计: {complexity_stats}")
        
        # 检查存储状态
        storage_stats = self._analyze_storage()
        
        # 系统信息
        system_info = {
            'working_dir': self.rag.working_dir,
            'complexity_stats': complexity_stats,
            'storage_stats': storage_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存分析结果
        analysis_file = os.path.join(self.output_dir, 'performance_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, ensure_ascii=False, indent=2)
        
        print(f"性能分析结果已保存到: {analysis_file}")
        return system_info
    
    def _analyze_storage(self) -> Dict[str, Any]:
        """分析存储状态"""
        working_dir = self.rag.working_dir
        storage_stats = {}
        
        files_to_check = [
            "vdb_chunks.json",
            "vdb_entities.json", 
            "kv_store_text_chunks.json",
            "kv_store_entities.json",
            "kv_store_community_reports.json",
            "graph_chunk_entity_relation.graphml"
        ]
        
        print(f"\n存储状态:")
        print(f"工作目录: {working_dir}")
        
        total_size = 0
        for file_name in files_to_check:
            file_path = os.path.join(working_dir, file_name)
            exists = os.path.exists(file_path)
            size = os.path.getsize(file_path) if exists else 0
            total_size += size
            
            status = '✅' if exists else '❌'
            print(f"  {file_name}: {status} ({size} bytes)")
            
            storage_stats[file_name] = {
                'exists': exists,
                'size': size
            }
        
        storage_stats['total_size'] = total_size
        print(f"总存储大小: {total_size} bytes")
        
        return storage_stats
    
    def save_test_results(self, results: List[Dict[str, Any]], filename: str) -> None:
        """保存测试结果"""
        results_file = os.path.join(self.output_dir, filename)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存到: {results_file}")


#------------------------------------------------------------------------------
# 主函数
#------------------------------------------------------------------------------

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级复杂度分类器RAG流程演示')
    parser.add_argument('--docs_path', type=str, default='./sample_docs',
                       help='文档路径（文件或目录）')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--working_dir', type=str, default='./complexity_classifier_cache',
                       help='工作目录')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='复杂度分类置信度阈值')
    
    args = parser.parse_args()
    
    print("🚀 开始高级复杂度分类器RAG流程演示")
    print("="*60)
    
    # 1. 加载文档
    print(f"\n📝 步骤1: 从 {args.docs_path} 加载文档")
    documents = DocumentLoader.load_documents(args.docs_path)
    print(f"加载了 {len(documents)} 个文档")
    
    if not documents:
        print("⚠️ 没有加载到文档，使用示例文档")
        # 使用示例文档
        sample_docs = [
            "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是AI的核心技术之一，它使计算机能够从数据中学习并做出预测或决策。",
            "深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程。"
        ]
        documents = sample_docs
    
    # 2. 配置复杂度分类器
    print(f"\n⚙️ 步骤2: 配置复杂度分类器")
    config = ComplexityClassifierConfig(
        confidence_threshold=args.confidence_threshold
    )
    
    # 3. 构建RAG系统
    print(f"\n🔧 步骤3: 构建RAG系统")
    builder = RAGSystemBuilder(args.working_dir, config)
    rag = builder.build()
    
    # 4. 插入文档
    print(f"\n📚 步骤4: 插入文档")
    builder.insert_documents(documents)
    
    # 5. 测试查询
    print(f"\n🔍 步骤5: 测试查询")
    tester = QueryTester(rag)
    
    # 测试查询集
    test_queries = [
        "什么是人工智能？",
        "机器学习有哪些算法？",
        "深度学习和机器学习的关系是什么？",
        "神经网络的工作原理是什么？"
    ]
    
    # 测试复杂度分类
    complexity_results = await tester.test_complexity_classification(test_queries)
    
    # 测试复杂度路由查询
    routing_results = await tester.test_queries_with_routing(test_queries)
    
    # 测试不同模式
    mode_results = tester.test_different_modes("深度学习和机器学习的关系是什么？")
    
    # 6. 性能分析
    print(f"\n📊 步骤6: 性能分析")
    analyzer = PerformanceAnalyzer(rag, args.output_dir)
    performance_results = analyzer.analyze_system_performance()
    
    # 保存结果
    analyzer.save_test_results(complexity_results, 'complexity_classification_results.json')
    analyzer.save_test_results(routing_results, 'routing_test_results.json')
    analyzer.save_test_results(mode_results, 'mode_comparison_results.json')
    
    print("\n" + "="*60)
    print("🎉 高级复杂度分类器RAG流程演示完成！")
    print("="*60)
    
    # 输出摘要
    print(f"\n📋 结果摘要:")
    print(f"- 处理文档数: {len(documents)}")
    print(f"- 测试查询数: {len(test_queries)}")
    print(f"- 复杂度分类准确率: {sum(1 for r in complexity_results if r['confidence'] > 0.7) / len(complexity_results):.1%}")
    print(f"- 查询成功率: {sum(1 for r in routing_results if r['success']) / len(routing_results):.1%}")
    print(f"- 平均查询时间: {sum(r['elapsed_time'] for r in routing_results) / len(routing_results):.2f}秒")
    print(f"- 结果保存位置: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 