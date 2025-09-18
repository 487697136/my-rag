"""
评估指标计算模块

包含以下功能:
1. ClassificationMetrics: 分类性能指标
2. CalibrationMetrics: 校准质量指标  
3. RoutingMetrics: 路由性能指标
4. StatisticalTests: 统计显著性检验
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    log_loss, brier_score_loss
)
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """分类性能指标计算器"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or ['zero_hop', 'one_hop', 'multi_hop']
        self.num_classes = len(self.class_names)
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """计算所有分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            
        Returns:
            包含所有指标的字典
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )
        
        # 各类别指标
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
            metrics[f'{class_name}_support'] = support[i]
        
        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro'
        )
        
        metrics['macro_precision'] = macro_precision
        metrics['macro_recall'] = macro_recall
        metrics['macro_f1'] = macro_f1
        metrics['micro_precision'] = micro_precision
        metrics['micro_recall'] = micro_recall
        metrics['micro_f1'] = micro_f1
        
        # 类别间方差
        metrics['f1_variance'] = np.var(f1)
        metrics['f1_std'] = np.std(f1)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # 如果有概率预测，计算相关指标
        if y_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_proba)
            
            # 每类别的Brier分数
            brier_scores = []
            for i in range(self.num_classes):
                y_binary = (y_true == i).astype(int)
                brier_score = brier_score_loss(y_binary, y_proba[:, i])
                brier_scores.append(brier_score)
                metrics[f'{self.class_names[i]}_brier'] = brier_score
            
            metrics['mean_brier_score'] = np.mean(brier_scores)
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """计算每个类别的详细指标"""
        
        report_dict = classification_report(
            y_true, y_pred,
            labels=range(self.num_classes),
            target_names=self.class_names,
            output_dict=True
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(report_dict).T
        
        # 添加支持度百分比
        total_support = df.loc[self.class_names, 'support'].sum()
        df['support_pct'] = df['support'] / total_support * 100
        
        return df
    
    def format_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> pd.DataFrame:
        """格式化混淆矩阵为DataFrame"""
        
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        df = pd.DataFrame(
            cm,
            index=[f'True_{name}' for name in self.class_names],
            columns=[f'Pred_{name}' for name in self.class_names]
        )
        
        return df


class CalibrationMetrics:
    """校准质量指标计算器
    
    实现ECE、MCE、Brier Score等校准指标，
    这些是本实验的核心评估指标。
    """
    
    def __init__(self, num_bins: int = 15, bin_strategy: str = 'equal_width'):
        """
        Args:
            num_bins: 可靠性图的bin数量
            bin_strategy: 分箱策略（'equal_width' 或 'equal_frequency'）
        """
        self.num_bins = num_bins
        self.bin_strategy = bin_strategy
    
    def compute_all_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        return_reliability_data: bool = True
    ) -> Dict[str, Any]:
        """计算所有校准指标
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            return_reliability_data: 是否返回可靠性图数据
            
        Returns:
            包含所有校准指标的字典
        """
        metrics = {}
        
        # ECE (Expected Calibration Error) - 核心指标
        ece, reliability_data = self._compute_ece(
            y_true,
            y_proba,
            bin_strategy=self.bin_strategy,
            return_data=True
        )
        metrics['ECE'] = ece
        
        # MCE (Maximum Calibration Error)
        metrics['MCE'] = self._compute_mce(y_true, y_proba)
        
        # Brier Score
        metrics['brier_score'] = self._compute_brier_score(y_true, y_proba)
        
        # Negative Log Likelihood
        metrics['nll'] = self._compute_nll(y_true, y_proba)
        
        # 置信度分布统计
        confidences = np.max(y_proba, axis=1)
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        
        # 准确率
        predictions = np.argmax(y_proba, axis=1)
        metrics['accuracy'] = accuracy_score(y_true, predictions)
        
        # 过自信和欠自信指标
        overconfidence, underconfidence = self._compute_confidence_bias(y_true, y_proba)
        metrics['overconfidence'] = overconfidence
        metrics['underconfidence'] = underconfidence
        
        # 可靠性图数据
        if return_reliability_data:
            metrics['reliability_data'] = reliability_data
        
        return metrics
    
    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        bin_strategy: str = 'equal_width',
        return_data: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """计算Expected Calibration Error (ECE)
        
        这是本实验的核心指标，目标是 < 0.08
        """
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        
        if bin_strategy == 'equal_width':
            bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        elif bin_strategy == 'equal_frequency':
            bin_boundaries = np.percentile(confidences, np.linspace(0, 100, self.num_bins + 1))
        else:
            raise ValueError(f"未知的分箱策略: {bin_strategy}")
        
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        reliability_data = {
            'bin_centers': [],
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': [],
            'bin_errors': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 确定当前bin中的样本
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += bin_error * prop_in_bin
                
                # 记录可靠性图数据
                reliability_data['bin_centers'].append((bin_lower + bin_upper) / 2)
                reliability_data['bin_accuracies'].append(accuracy_in_bin)
                reliability_data['bin_confidences'].append(avg_confidence_in_bin)
                reliability_data['bin_counts'].append(in_bin.sum())
                reliability_data['bin_errors'].append(bin_error)
            else:
                # 空bin
                reliability_data['bin_centers'].append((bin_lower + bin_upper) / 2)
                reliability_data['bin_accuracies'].append(0)
                reliability_data['bin_confidences'].append(0)
                reliability_data['bin_counts'].append(0)
                reliability_data['bin_errors'].append(0)
        
        if return_data:
            return ece, reliability_data
        return ece
    
    def _compute_mce(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """计算Maximum Calibration Error (MCE)"""
        _, reliability_data = self._compute_ece(
            y_true, y_proba, bin_strategy=self.bin_strategy, return_data=True
        )
        
        bin_errors = reliability_data['bin_errors']
        return max(bin_errors) if bin_errors else 0.0
    
    def _compute_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """计算Brier Score"""
        n_classes = y_proba.shape[1]
        one_hot = np.eye(n_classes)[y_true]
        return np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))
    
    def _compute_nll(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """计算Negative Log Likelihood"""
        # 避免log(0)
        y_proba_clipped = np.clip(y_proba, 1e-8, 1 - 1e-8)
        return -np.mean(np.log(y_proba_clipped[np.arange(len(y_true)), y_true]))
    
    def _compute_confidence_bias(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[float, float]:
        """计算过自信和欠自信指标"""
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        
        # 过自信：置信度 > 准确率的程度
        overconfident_mask = confidences > accuracies
        overconfidence = np.mean(confidences[overconfident_mask] - accuracies[overconfident_mask]) if overconfident_mask.any() else 0
        
        # 欠自信：准确率 > 置信度的程度
        underconfident_mask = confidences < accuracies
        underconfidence = np.mean(accuracies[underconfident_mask] - confidences[underconfident_mask]) if underconfident_mask.any() else 0
        
        return overconfidence, underconfidence
    
    def compare_calibration_methods(
        self,
        y_true: np.ndarray,
        method_probabilities: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """比较不同校准方法的效果"""
        
        results = []
        
        for method_name, y_proba in method_probabilities.items():
            metrics = self.compute_all_calibration_metrics(
                y_true, y_proba, return_reliability_data=False
            )
            
            result = {
                'method': method_name,
                'ECE': metrics['ECE'],
                'MCE': metrics['MCE'],
                'brier_score': metrics['brier_score'],
                'nll': metrics['nll'],
                'accuracy': metrics['accuracy'],
                'mean_confidence': metrics['mean_confidence']
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('ECE')  # 按ECE排序
        
        return df


class RoutingMetrics:
    """路由性能指标计算器
    
    评估复杂度分类器在路由决策中的表现。
    """
    
    def __init__(self, route_mapping: Optional[Dict] = None):
        """
        Args:
            route_mapping: 复杂度到路由模式的映射
        """
        self.route_mapping = route_mapping or {
            0: 'llm_only',     # zero_hop
            1: 'naive',        # one_hop  
            2: 'local'         # multi_hop (简化，实际可能更复杂)
        }
    
    def compute_route_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算路由准确率
        
        路由准确率 = 正确分类的查询比例
        目标: > 90%
        """
        return accuracy_score(y_true, y_pred)
    
    def compute_optimal_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> Dict[str, float]:
        """计算最优覆盖率
        
        在不同置信度阈值下，选择最优检索模式的覆盖率。
        """
        confidences = np.max(y_proba, axis=1)
        high_confidence_mask = confidences >= confidence_threshold
        
        # 高置信度样本的准确率
        if high_confidence_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                y_true[high_confidence_mask],
                y_pred[high_confidence_mask]
            )
            coverage = high_confidence_mask.mean()
        else:
            high_conf_accuracy = 0.0
            coverage = 0.0
        
        return {
            'optimal_coverage': coverage,
            'high_confidence_accuracy': high_conf_accuracy,
            'confidence_threshold': confidence_threshold
        }
    
    def analyze_route_distribution(
        self,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """分析路由模式分布"""
        
        # 预测的复杂度分布
        complexity_counts = np.bincount(y_pred, minlength=3)
        total_queries = len(y_pred)
        
        distribution = {}
        for complexity, count in enumerate(complexity_counts):
            route_mode = self.route_mapping.get(complexity, f'mode_{complexity}')
            distribution[route_mode] = {
                'count': count,
                'percentage': count / total_queries * 100,
                'avg_confidence': np.mean(y_proba[y_pred == complexity, complexity]) if count > 0 else 0
            }
        
        return distribution
    
    def threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> pd.DataFrame:
        """不同置信度阈值下的性能分析"""
        
        results = []
        
        for threshold in thresholds:
            confidences = np.max(y_proba, axis=1)
            high_conf_mask = confidences >= threshold
            
            if high_conf_mask.sum() > 0:
                # 高置信度样本的指标
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true[high_conf_mask],
                    y_pred[high_conf_mask],
                    average='macro'
                )
                accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
                coverage = high_conf_mask.mean()
            else:
                precision = recall = f1 = accuracy = coverage = 0.0
            
            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'samples': high_conf_mask.sum()
            })
        
        return pd.DataFrame(results)


class StatisticalTests:
    """统计显著性检验
    
    用于验证模型改进的统计显著性。
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 显著性水平
        """
        self.alpha = alpha
    
    def paired_t_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """配对t检验
        
        比较两组配对样本的均值差异。
        """
        statistic, p_value = ttest_rel(scores1, scores2, alternative=alternative)
        
        # 计算效应大小 (Cohen's d)
        mean_diff = np.mean(scores1 - scores2)
        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'mean_diff': mean_diff,
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d)
        }
    
    def wilcoxon_signed_rank_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """Wilcoxon符号秩检验（非参数）"""
        
        try:
            statistic, p_value = wilcoxon(scores1, scores2, alternative=alternative)
        except ValueError as e:
            # 处理所有差值为0的情况
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'error': str(e)
            }
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Bootstrap置信区间"""
        
        np.random.seed(random_state)
        
        bootstrap_stats = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'statistic': statistic_func(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'bootstrap_std': np.std(bootstrap_stats)
        }
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """解释Cohen's d效应大小"""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Dict[str, Any]:
        """多重比较校正"""
        
        p_values = np.array(p_values)
        n_comparisons = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni校正
            corrected_alpha = self.alpha / n_comparisons
            significant = p_values < corrected_alpha
            corrected_p_values = np.minimum(p_values * n_comparisons, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni校正
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p_values = np.zeros_like(p_values)
            significant = np.zeros_like(p_values, dtype=bool)
            
            for i, (idx, p_val) in enumerate(zip(sorted_indices, sorted_p)):
                corrected_p = p_val * (n_comparisons - i)
                corrected_p_values[idx] = min(corrected_p, 1.0)
                significant[idx] = corrected_p < self.alpha
                
                # 如果当前不显著，后续都不显著
                if not significant[idx]:
                    break
        
        else:
            raise ValueError(f"未知的校正方法: {method}")
        
        return {
            'corrected_p_values': corrected_p_values,
            'significant': significant,
            'method': method,
            'n_comparisons': n_comparisons,
            'corrected_alpha': corrected_alpha if method == 'bonferroni' else self.alpha
        }
    
    def comprehensive_comparison(
        self,
        baseline_scores: np.ndarray,
        method_scores: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """全面的方法比较分析"""
        
        results = []
        
        for method_name, scores in method_scores.items():
            # 配对t检验
            t_test_result = self.paired_t_test(scores, baseline_scores)
            
            # Wilcoxon检验
            wilcoxon_result = self.wilcoxon_signed_rank_test(scores, baseline_scores)
            
            # Bootstrap置信区间
            mean_diff = scores - baseline_scores
            bootstrap_result = self.bootstrap_confidence_interval(mean_diff)
            
            result = {
                'method': method_name,
                'mean_score': np.mean(scores),
                'baseline_mean': np.mean(baseline_scores),
                'mean_difference': np.mean(mean_diff),
                't_test_p_value': t_test_result['p_value'],
                't_test_significant': t_test_result['significant'],
                'cohens_d': t_test_result['cohens_d'],
                'effect_size': t_test_result['effect_size_interpretation'],
                'wilcoxon_p_value': wilcoxon_result['p_value'],
                'wilcoxon_significant': wilcoxon_result['significant'],
                'ci_lower': bootstrap_result['ci_lower'],
                'ci_upper': bootstrap_result['ci_upper']
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # 多重比较校正
        p_values = df['t_test_p_value'].values
        correction_result = self.multiple_comparison_correction(p_values)
        
        df['corrected_p_value'] = correction_result['corrected_p_values']
        df['corrected_significant'] = correction_result['significant']
        
        return df