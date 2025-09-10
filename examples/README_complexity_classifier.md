# 复杂度分类器使用指南

本指南介绍如何使用nano-graphrag的复杂度分类器进行完整的RAG流程。

## 🚀 快速开始

### 1. 基础示例

运行基础示例，使用内置的示例文档：

```bash
python examples/using_complexity_classifier.py
```

这个示例会：
- 加载示例文档（AI相关主题）
- 构建知识图谱和向量数据库
- 测试复杂度分类器
- 演示自适应查询路由

### 2. 高级示例

运行高级示例，支持自定义文档和详细分析：

```bash
# 使用默认参数
python examples/using_complexity_classifier_advanced.py

# 指定文档路径
python examples/using_complexity_classifier_advanced.py --docs_path ./your_documents/

# 指定输出目录
python examples/using_complexity_classifier_advanced.py --output_dir ./results/

# 调整置信度阈值
python examples/using_complexity_classifier_advanced.py --confidence_threshold 0.7
```

## 📁 支持的文档格式

高级示例支持以下文档格式：

### 1. TXT文件
```
人工智能是计算机科学的一个分支...
机器学习是AI的核心技术之一...
```

### 2. JSON文件
```json
[
  {
    "title": "文档标题",
    "content": "文档内容..."
  },
  {
    "text": "另一个文档内容..."
  }
]
```

### 3. JSONL文件
```jsonl
{"content": "第一个文档内容..."}
{"text": "第二个文档内容..."}
{"title": "标题", "content": "内容..."}
```

## 🔧 复杂度分类器配置

### 配置参数

```python
from nano_graphrag.complexity import ComplexityAwareRouter

router = ComplexityAwareRouter(
    model_path="nano_graphrag/models/modernbert_complexity_classifier",
    confidence_threshold=0.6,  # 置信度阈值
    enable_fallback=True,      # 启用规则回退
    use_modernbert=True        # 使用ModernBERT模型
)
```

### 复杂度等级

- **zero_hop**: 常识性问题，无需检索
  - 路由到: `llm_only` 模式
  - 示例: "什么是人工智能？", "2+2等于多少？"

- **one_hop**: 单步检索问题
  - 路由到: `naive` 或 `bm25` 模式
  - 示例: "机器学习有哪些算法？", "PyTorch是什么？"

- **multi_hop**: 多步推理问题
  - 路由到: `local` 或 `global` 模式
  - 示例: "深度学习和机器学习的关系是什么？"

## 🔍 查询模式

### 1. 复杂度感知查询（推荐）

```python
# 自动根据查询复杂度选择最佳模式
response = await rag.aquery("你的查询")
```

### 2. 手动指定模式

```python
from nano_graphrag import QueryParam

# 直接LLM回答
response = rag.query("查询", param=QueryParam(mode="llm_only"))

# 向量检索
response = rag.query("查询", param=QueryParam(mode="naive"))

# 关键词检索
response = rag.query("查询", param=QueryParam(mode="bm25"))

# 局部图推理
response = rag.query("查询", param=QueryParam(mode="local"))

# 全局图推理
response = rag.query("查询", param=QueryParam(mode="global"))
```

## 📊 性能分析

运行高级示例后，会在输出目录生成以下分析文件：

- `performance_analysis.json`: 系统性能分析
- `complexity_classification_results.json`: 复杂度分类结果
- `routing_test_results.json`: 路由测试结果
- `mode_comparison_results.json`: 模式对比结果

### 性能指标

- **复杂度分类准确率**: 模型预测的置信度
- **查询成功率**: 成功回答的查询比例
- **平均查询时间**: 每个查询的平均处理时间
- **存储状态**: 各种缓存文件的大小和状态

## 🛠️ 自定义使用

### 1. 创建自定义RAG系统

```python
from nano_graphrag import GraphRAG
from nano_graphrag.complexity import ComplexityAwareRouter

# 创建复杂度感知路由器
router = ComplexityAwareRouter(
    model_path="your_model_path",
    confidence_threshold=0.7
)

# 创建RAG系统
rag = GraphRAG(
    working_dir="./your_cache",
    enable_naive_rag=True,
    enable_bm25_rag=True,
    enable_graph_rag=True,
    router=router
)

# 插入文档
rag.insert("你的文档内容")

# 查询
response = await rag.aquery("你的查询")
```

### 2. 批量处理文档

```python
import os
from pathlib import Path

# 加载文档
documents = []
docs_dir = Path("./your_documents")
for file_path in docs_dir.glob("*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents.append(f.read())

# 插入文档
for doc in documents:
    rag.insert(doc)
```

### 3. 测试复杂度分类

```python
# 测试单个查询的复杂度
complexity_result = await router.predict_complexity_detailed("你的查询")
print(f"复杂度: {complexity_result['complexity']}")
print(f"置信度: {complexity_result['confidence']}")
print(f"候选模式: {complexity_result['candidate_modes']}")
```

## 🔧 故障排除

### 1. 模型加载失败

如果ModernBERT模型加载失败，系统会自动回退到规则分类：

```python
# 检查模型路径
model_path = "nano_graphrag/models/modernbert_complexity_classifier"
if not os.path.exists(model_path):
    print("模型路径不存在，将使用规则分类")
```

### 2. 查询失败

如果查询失败，检查：

- 是否已插入文档
- 网络连接是否正常（如果使用在线LLM）
- 工作目录是否有写入权限

### 3. 性能问题

如果性能较慢：

- 减少文档数量
- 调整置信度阈值
- 使用更快的LLM服务

## 📈 最佳实践

### 1. 文档准备

- 确保文档内容清晰、结构化
- 避免过长的文档，建议分段处理
- 包含丰富的实体和关系信息

### 2. 查询优化

- 使用清晰、具体的问题
- 避免过于复杂或模糊的查询
- 根据查询类型选择合适的模式

### 3. 系统配置

- 根据硬件配置调整模型参数
- 定期清理缓存文件
- 监控系统性能指标

## 🎯 使用场景

### 1. 知识问答系统

适用于构建基于文档的智能问答系统，自动选择最佳检索策略。

### 2. 文档检索系统

支持多种检索模式，提供更准确的文档检索结果。

### 3. 研究助手

帮助研究人员快速从大量文档中提取相关信息。

### 4. 客服系统

构建智能客服系统，提供准确、快速的回答。

## 📞 技术支持

如果遇到问题，请：

1. 检查日志输出
2. 查看性能分析结果
3. 参考示例代码
4. 提交Issue到项目仓库

---

**祝您使用愉快！** 🎉 