"""
数据加载与处理模块

包含以下功能:
1. ComplexityDataLoader: 复杂度数据集加载器
2. DatasetSampler: 数据集采样器
3. QueryComplexityLabeler: 查询复杂度标注器
"""

import json
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import logging

logger = logging.getLogger(__name__)


class ComplexityDataLoader:
    """复杂度数据集加载器
    
    负责从多个数据源（MS MARCO, Natural Questions, HotpotQA）
    加载和统一处理查询复杂度数据。
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Args:
            data_dir: 数据目录路径
            config: 数据加载配置
            random_state: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.config = config or self._default_config()
        self.random_state = random_state
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 数据缓存
        self._dataset_cache = {}
        self._complexity_mapping = {
            'zero_hop': 0,
            'one_hop': 1, 
            'multi_hop': 2
        }
        
    def _default_config(self) -> Dict:
        """默认数据配置"""
        return {
            'datasets': {
                'ms_marco': {
                    'file_pattern': 'ms_marco*.jsonl',
                    'sample_size': 4000,
                    'query_field': 'query',
                    'answer_field': 'answer',
                    'complexity_field': 'complexity',
                    'weight': 1.0
                },
                'natural_questions': {
                    'file_pattern': 'natural_questions*.jsonl',
                    'sample_size': 4000,
                    'query_field': 'question',
                    'answer_field': 'answer',
                    'complexity_field': 'complexity',
                    'weight': 1.0
                },
                'hotpot_qa': {
                    'file_pattern': 'hotpot_qa*.jsonl',
                    'sample_size': 4000,
                    'query_field': 'question',
                    'answer_field': 'answer',
                    'complexity_field': 'complexity',
                    'weight': 1.0
                }
            },
            'complexity_distribution': {
                'zero_hop': 0.25,
                'one_hop': 0.40,
                'multi_hop': 0.35
            },
            'manual_review': {
                'sample_size': 500,
                'review_ratio': 0.1
            },
            'quality_filters': {
                'min_query_length': 3,
                'max_query_length': 200,
                'min_answer_length': 1,
                'require_answer': True
            }
        }
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """加载所有配置的数据集"""
        logger.info("开始加载所有数据集")
        
        all_datasets = {}
        
        for dataset_name, dataset_config in self.config['datasets'].items():
            try:
                dataset = self.load_single_dataset(dataset_name, dataset_config)
                all_datasets[dataset_name] = dataset
                logger.info(f"成功加载 {dataset_name}: {len(dataset)} 条样本")
            except Exception as e:
                logger.error(f"加载 {dataset_name} 失败: {e}")
                # 创建空数据集避免中断
                all_datasets[dataset_name] = pd.DataFrame()
        
        return all_datasets
    
    def load_single_dataset(self, dataset_name: str, config: Dict) -> pd.DataFrame:
        """加载单个数据集"""
        
        # 查找数据文件
        pattern = config['file_pattern']
        data_files = list(self.data_dir.glob(pattern))
        
        if not data_files:
            logger.warning(f"未找到 {dataset_name} 数据文件，模式: {pattern}")
            return pd.DataFrame()
        
        # 加载数据
        all_data = []
        for file_path in data_files:
            if file_path.suffix == '.jsonl':
                data = self._load_jsonl(file_path)
            elif file_path.suffix == '.json':
                data = self._load_json(file_path)
            elif file_path.suffix == '.csv':
                data = self._load_csv(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                continue
            
            all_data.extend(data)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        
        # 字段映射和标准化
        df = self._standardize_fields(df, config)
        
        # 应用质量过滤器
        df = self._apply_quality_filters(df)
        
        # 采样到目标大小
        target_size = config.get('sample_size', len(df))
        if len(df) > target_size:
            df = df.sample(n=target_size, random_state=self.random_state)
        
        # 添加数据集标识
        df['dataset'] = dataset_name
        
        return df
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSONL行失败: {e}")
        return data
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保返回列表格式
        if isinstance(data, dict):
            # 可能是带有'data'键的格式
            if 'data' in data:
                return data['data']
            elif 'examples' in data:
                return data['examples']
            else:
                return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """加载CSV文件"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _standardize_fields(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """标准化字段名称和格式"""
        field_mapping = {
            config.get('query_field', 'query'): 'query',
            config.get('answer_field', 'answer'): 'answer',
            config.get('complexity_field', 'complexity'): 'complexity'
        }
        
        # 重命名字段
        df = df.rename(columns=field_mapping)
        
        # 确保必需字段存在
        required_fields = ['query', 'complexity']
        for field in required_fields:
            if field not in df.columns:
                logger.warning(f"缺少必需字段: {field}")
                if field == 'complexity':
                    df[field] = 'unknown'  # 默认复杂度
                elif field == 'query':
                    df[field] = ''  # 空查询
        
        # 标准化复杂度标签
        df['complexity'] = df['complexity'].apply(self._normalize_complexity_label)
        
        return df
    
    def _normalize_complexity_label(self, label: str) -> str:
        """标准化复杂度标签"""
        if pd.isna(label):
            return 'unknown'
        
        label = str(label).lower().strip()
        
        # 映射规则
        if label in ['0', 'zero', 'zero_hop', 'zero-hop', 'simple']:
            return 'zero_hop'
        elif label in ['1', 'one', 'one_hop', 'one-hop', 'medium']:
            return 'one_hop'
        elif label in ['2', 'two', 'multi', 'multi_hop', 'multi-hop', 'complex']:
            return 'multi_hop'
        else:
            return 'unknown'
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数据质量过滤器"""
        initial_size = len(df)
        filters = self.config.get('quality_filters', {})
        
        # 查询长度过滤
        if 'min_query_length' in filters:
            min_len = filters['min_query_length']
            df = df[df['query'].str.split().str.len() >= min_len]
        
        if 'max_query_length' in filters:
            max_len = filters['max_query_length']
            df = df[df['query'].str.split().str.len() <= max_len]
        
        # 答案要求过滤
        if filters.get('require_answer', False):
            df = df[df['answer'].notna() & (df['answer'].str.strip() != '')]
        
        # 复杂度标签过滤
        df = df[df['complexity'].isin(['zero_hop', 'one_hop', 'multi_hop'])]
        
        filtered_size = len(df)
        if filtered_size < initial_size:
            logger.info(f"质量过滤: {initial_size} -> {filtered_size} 条样本")
        
        return df
    
    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并多个数据集"""
        logger.info("合并数据集")
        
        combined_data = []
        for dataset_name, df in datasets.items():
            if not df.empty:
                combined_data.append(df)
        
        if not combined_data:
            logger.error("没有有效数据集可合并")
            return pd.DataFrame()
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # 打乱数据
        combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"合并完成，总计 {len(combined_df)} 条样本")
        self._log_dataset_statistics(combined_df)
        
        return combined_df
    
    def _log_dataset_statistics(self, df: pd.DataFrame):
        """记录数据集统计信息"""
        logger.info("数据集统计信息:")
        
        # 复杂度分布
        complexity_counts = df['complexity'].value_counts()
        total_samples = len(df)
        
        for complexity, count in complexity_counts.items():
            ratio = count / total_samples
            logger.info(f"  {complexity}: {count} ({ratio:.2%})")
        
        # 数据集来源分布
        if 'dataset' in df.columns:
            dataset_counts = df['dataset'].value_counts()
            for dataset, count in dataset_counts.items():
                ratio = count / total_samples
                logger.info(f"  {dataset}: {count} ({ratio:.2%})")


class DatasetSampler:
    """数据集采样器
    
    负责按指定分布采样数据，确保复杂度比例符合实验要求。
    """
    
    def __init__(
        self,
        target_distribution: Optional[Dict[str, float]] = None,
        random_state: int = 42
    ):
        """
        Args:
            target_distribution: 目标复杂度分布
            random_state: 随机种子
        """
        self.target_distribution = target_distribution or {
            'zero_hop': 0.25,
            'one_hop': 0.40,
            'multi_hop': 0.35
        }
        self.random_state = random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def stratified_sample(
        self,
        df: pd.DataFrame,
        total_size: int,
        complexity_column: str = 'complexity'
    ) -> pd.DataFrame:
        """按复杂度分层采样"""
        logger.info(f"按目标分布采样 {total_size} 条样本")
        
        sampled_dfs = []
        
        for complexity, target_ratio in self.target_distribution.items():
            target_count = int(total_size * target_ratio)
            
            # 获取该复杂度的所有样本
            complexity_df = df[df[complexity_column] == complexity]
            
            if len(complexity_df) == 0:
                logger.warning(f"没有 {complexity} 复杂度的样本")
                continue
            
            # 采样
            if len(complexity_df) >= target_count:
                sampled_df = complexity_df.sample(n=target_count, random_state=self.random_state)
            else:
                logger.warning(f"{complexity} 样本不足: 需要 {target_count}, 实际 {len(complexity_df)}")
                sampled_df = complexity_df  # 使用所有可用样本
            
            sampled_dfs.append(sampled_df)
            logger.info(f"  {complexity}: 采样 {len(sampled_df)} 条")
        
        # 合并采样结果
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # 打乱顺序
        result_df = result_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"分层采样完成，共 {len(result_df)} 条样本")
        return result_df
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.9,
        test_size: Optional[int] = None,
        stratify_column: str = 'complexity'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分层划分训练集和测试集"""
        
        if test_size is not None:
            # 固定测试集大小
            test_ratio = test_size / len(df)
            train_ratio = 1 - test_ratio
        else:
            test_ratio = 1 - train_ratio
        
        logger.info(f"划分数据集: 训练集 {train_ratio:.2%}, 测试集 {test_ratio:.2%}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df[stratify_column],
            random_state=self.random_state
        )
        
        logger.info(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
        
        return train_df, test_df
    
    def calibration_split(
        self,
        train_df: pd.DataFrame,
        calibration_ratio: float = 0.1,
        stratify_column: str = 'complexity'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """从训练集中划分校准集"""
        
        logger.info(f"从训练集划分校准集: {calibration_ratio:.2%}")
        
        train_final, cal_df = train_test_split(
            train_df,
            test_size=calibration_ratio,
            stratify=train_df[stratify_column],
            random_state=self.random_state
        )
        
        logger.info(f"最终训练集: {len(train_final)} 条, 校准集: {len(cal_df)} 条")
        
        return train_final, cal_df


class QueryComplexityLabeler:
    """查询复杂度标注器
    
    提供半自动化的复杂度标注功能，用于人工审核和标签修正。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 标注配置
        """
        self.config = config or self._default_config()
        self.annotation_history = []
    
    def _default_config(self) -> Dict:
        """默认标注配置"""
        return {
            'zero_hop_patterns': [
                r'\bwhat is\b', r'\bdefine\b', r'\bmeaning\b',
                r'\bwho is\b', r'\bwhen was\b', r'\bwhere is\b'
            ],
            'one_hop_patterns': [
                r'\bwho\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b',
                r'\bhow many\b', r'\blist\b', r'\bname\b'
            ],
            'multi_hop_patterns': [
                r'\bcompare\b', r'\brelationship\b', r'\bboth\b',
                r'\band\b.*\bor\b', r'\bdifference\b', r'\bsimilar\b'
            ],
            'length_thresholds': {
                'short': 5,
                'medium': 15,
                'long': 25
            }
        }
    
    def auto_label_batch(self, queries: List[str]) -> List[str]:
        """批量自动标注查询复杂度"""
        import re
        
        labels = []
        for query in queries:
            label = self._auto_label_single(query)
            labels.append(label)
        
        return labels
    
    def _auto_label_single(self, query: str) -> str:
        """自动标注单个查询"""
        import re
        
        query_lower = query.lower()
        
        # 检查模式匹配
        zero_hop_matches = sum(1 for pattern in self.config['zero_hop_patterns']
                              if re.search(pattern, query_lower))
        one_hop_matches = sum(1 for pattern in self.config['one_hop_patterns']
                             if re.search(pattern, query_lower))
        multi_hop_matches = sum(1 for pattern in self.config['multi_hop_patterns']
                               if re.search(pattern, query_lower))
        
        # 基于匹配数量判断
        if multi_hop_matches > 0:
            return 'multi_hop'
        elif one_hop_matches > 0:
            return 'one_hop'
        elif zero_hop_matches > 0:
            return 'zero_hop'
        
        # 基于长度判断
        word_count = len(query.split())
        if word_count <= self.config['length_thresholds']['short']:
            return 'zero_hop'
        elif word_count <= self.config['length_thresholds']['medium']:
            return 'one_hop'
        else:
            return 'multi_hop'
    
    def manual_review_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 500,
        complexity_column: str = 'complexity'
    ) -> pd.DataFrame:
        """人工审核样本（模拟）"""
        
        logger.info(f"模拟人工审核 {sample_size} 条样本")
        
        # 随机选择审核样本
        review_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # 模拟审核过程（添加一些修正）
        corrected_df = review_sample.copy()
        
        # 模拟10%的标签需要修正
        correction_indices = np.random.choice(
            len(corrected_df),
            size=int(len(corrected_df) * 0.1),
            replace=False
        )
        
        complexity_options = ['zero_hop', 'one_hop', 'multi_hop']
        for idx in correction_indices:
            current_label = corrected_df.iloc[idx][complexity_column]
            # 随机选择一个不同的标签
            new_options = [label for label in complexity_options if label != current_label]
            new_label = np.random.choice(new_options)
            corrected_df.iloc[idx, corrected_df.columns.get_loc(complexity_column)] = new_label
        
        logger.info(f"模拟修正了 {len(correction_indices)} 条标签")
        
        return corrected_df
    
    def validate_annotations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证标注质量"""
        
        validation_results = {
            'total_samples': len(df),
            'complexity_distribution': df['complexity'].value_counts().to_dict(),
            'missing_labels': df['complexity'].isna().sum(),
            'unknown_labels': (df['complexity'] == 'unknown').sum(),
            'quality_score': 0.0
        }
        
        # 计算质量分数
        valid_labels = df['complexity'].isin(['zero_hop', 'one_hop', 'multi_hop']).sum()
        validation_results['quality_score'] = valid_labels / len(df)
        
        return validation_results