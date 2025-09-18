#!/usr/bin/env python3
"""
实验一：数据准备脚本

功能:
1. 使用项目中已有的训练数据和复杂度分类器
2. 从training_data目录加载真实标注数据
3. 准备校准和测试数据集
4. 验证现有模型的性能

运行方式:
    python scripts/01_data_preparation.py
"""

import os
import sys
import yaml
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心库
# 注意：为保证“真实数据优先”，本脚本不再进行任何合成/模拟数据生成

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_output_directories():
    """准备输出目录"""
    directories = [
        'outputs/logs',
        'outputs/models',
        'outputs/results',
        'outputs/figures',
        'data/processed'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")


def check_training_data(training_data_dir: Path) -> dict:
    """检查training_data目录中的真实数据"""
    logger.info("检查训练数据可用性...")
    
    data_status = {
        'labeled_queries': [],
        'labeled_data_file': None,
        'total_queries': 0
    }
    
    if not training_data_dir.exists():
        logger.warning(f"训练数据目录不存在: {training_data_dir}")
        return data_status
    
    # 查找标注数据文件
    patterns = ['labeled_*.json', 'labeled_*.jsonl', 'queries_*.json', 'complexity_*.json']
    
    found_files = []
    for pattern in patterns:
        found_files.extend(list(training_data_dir.glob(pattern)))
    
    if found_files:
        data_status['labeled_data_file'] = found_files[0]
        logger.info(f"找到标注数据文件: {found_files[0]}")
        
        # 尝试加载数据
        try:
            with open(found_files[0], 'r', encoding='utf-8') as f:
                if found_files[0].suffix == '.jsonl':
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            
            data_status['labeled_queries'] = data
            data_status['total_queries'] = len(data)
            logger.info(f"成功加载 {len(data)} 条标注查询")
            
        except Exception as e:
            logger.error(f"加载标注数据失败: {e}")
    
    else:
        logger.warning("未找到标注数据文件")
    
    return data_status


# 已移除分类器加载逻辑（本脚本仅做真实数据集准备，不再生成任何合成/模拟查询）


# 已移除：任何基于模板或模型的“测试查询生成”逻辑


def create_evaluation_datasets(labeled_data: list) -> dict:
    """创建评估数据集"""
    logger.info("创建评估数据集...")
    
    datasets = {}
    
    # 如果有真实标注数据，优先使用
    if labeled_data:
        logger.info(f"使用真实标注数据: {len(labeled_data)} 条")
        
        # 标准化数据格式
        standardized_data = []
        for item in labeled_data:
            # 处理不同的数据格式
            if 'query' in item and 'complexity' in item:
                standardized_data.append({
                    'query': item['query'],
                    'complexity': item['complexity'],
                    'source': 'labeled'
                })
            elif 'question' in item and 'complexity' in item:
                standardized_data.append({
                    'query': item['question'],
                    'complexity': item['complexity'],
                    'source': 'labeled'
                })
        
        # 划分数据集
        if len(standardized_data) >= 100:
            train_data, temp_data = train_test_split(
                standardized_data, test_size=0.4, random_state=42,
                stratify=[item['complexity'] for item in standardized_data]
            )
            cal_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=42,
                stratify=[item['complexity'] for item in temp_data]
            )
            
            datasets['train_data'] = train_data
            datasets['calibration_data'] = cal_data
            datasets['test_data'] = test_data
            
        else:
            # 数据太少，全部用于测试
            datasets['test_data'] = standardized_data
            datasets['train_data'] = []
            datasets['calibration_data'] = []
    
    # 不再使用任何生成的查询补充数据。若真实数据不足，则直接提示并保持数据为空，避免引入非真实数据。
    
    # 报告数据集大小
    for split, data in datasets.items():
        if data:
            complexity_dist = Counter(item['complexity'] for item in data)
            logger.info(f"{split}: {len(data)} 条, 分布: {dict(complexity_dist)}")
        else:
            logger.warning(f"{split}: 0 条数据")
    
    return datasets


def save_datasets(datasets: dict, output_dir: Path):
    """保存数据集"""
    logger.info("保存数据集...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in datasets.items():
        if not data:
            logger.warning(f"跳过空数据集: {split_name}")
            continue
            
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存CSV文件
        csv_file = output_dir / f'{split_name}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"保存 {split_name}: {csv_file} ({len(df)} 条)")
        
        # 保存JSON文件（备份）
        json_file = output_dir / f'{split_name}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def generate_data_report(datasets: dict, output_dir: Path):
    """生成数据报告"""
    logger.info("生成数据报告...")
    
    report = {
        'data_preparation_summary': {
            'total_datasets': len(datasets),
            'total_samples': sum(len(data) for data in datasets.values()),
            'datasets_info': {}
        },
        'complexity_distribution': {},
        'data_quality': {},
        'recommendations': []
    }
    
    # 数据集信息
    for split_name, data in datasets.items():
        if data:
            complexity_dist = Counter(item['complexity'] for item in data)
            source_dist = Counter(item.get('source', 'unknown') for item in data)
            
            report['datasets_info'][split_name] = {
                'size': len(data),
                'complexity_distribution': dict(complexity_dist),
                'source_distribution': dict(source_dist)
            }
            
            report['complexity_distribution'][split_name] = dict(complexity_dist)
    
    # 数据质量评估
    all_data = []
    for data in datasets.values():
        all_data.extend(data)
    
    if all_data:
        total_complexity_dist = Counter(item['complexity'] for item in all_data)
        total_source_dist = Counter(item.get('source', 'unknown') for item in all_data)
        
        report['data_quality'] = {
            'total_samples': len(all_data),
            'complexity_balance': dict(total_complexity_dist),
            'source_distribution': dict(total_source_dist),
            'has_real_data': 'labeled' in total_source_dist,
            'real_data_ratio': total_source_dist.get('labeled', 0) / len(all_data)
        }
        
        # 生成建议
        if report['data_quality']['real_data_ratio'] < 0.3:
            report['recommendations'].append("建议增加更多真实标注数据以提高实验可信度")
        
        if min(total_complexity_dist.values()) < max(total_complexity_dist.values()) * 0.3:
            report['recommendations'].append("复杂度类别分布不平衡，建议进行数据平衡处理")
        
        if len(all_data) < 1000:
            report['recommendations'].append("数据量较少，建议增加数据量以获得更稳定的实验结果")
    
    # 保存报告
    report_file = output_dir / 'data_preparation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存数据报告: {report_file}")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验一数据准备")
    parser.add_argument('--training-data-dir', type=str, 
                       default=str(project_root / 'training_data'),
                       help='训练数据目录路径')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='输出目录')
    args = parser.parse_args()
    
    logger.info("开始实验一数据准备...")
    
    # 准备输出目录
    prepare_output_directories()
    
    # 检查训练数据
    training_data_dir = Path(args.training_data_dir)
    data_status = check_training_data(training_data_dir)
    
    # 不再加载分类器用于生成查询；仅基于真实标注数据创建评估数据集
    datasets = create_evaluation_datasets(
        data_status['labeled_queries']
    )
    
    # 保存数据集
    output_dir = Path(args.output_dir)
    save_datasets(datasets, output_dir)
    
    # 生成数据报告
    report = generate_data_report(datasets, output_dir)
    
    # 输出摘要
    logger.info("\n=== 数据准备完成 ===")
    logger.info(f"使用真实标注数据: {data_status['total_queries']} 条")
    # 真实数据优先，不再统计生成测试查询
    
    for split_name, data in datasets.items():
        if data:
            logger.info(f"{split_name}: {len(data)} 条")
    
    logger.info(f"数据质量报告: {output_dir / 'data_preparation_report.json'}")
    
    # 检查关键问题
    if not datasets.get('test_data'):
        logger.error("没有测试数据，无法进行实验。请在 training_data/ 目录放置真实标注数据（含字段 query 与 complexity）。")
        return
    
    if len(datasets.get('test_data', [])) < 100:
        logger.warning("测试数据量较少，实验结果可能不够稳定")
    
    if report['data_quality'].get('real_data_ratio', 0) < 0.1:
        logger.warning("真实数据比例过低，建议增加标注数据")
    
    logger.info("数据准备完毕，可以开始后续实验!")


if __name__ == "__main__":
    main()