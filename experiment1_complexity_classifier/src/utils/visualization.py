"""
可视化工具模块

包含以下功能:
1. ReliabilityPlotter: 可靠性曲线绘制
2. PerformancePlotter: 性能对比图表
3. CalibrationVisualizer: 校准效果可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import logging

logger = logging.getLogger(__name__)

# 设置中文字体支持和绘图样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ReliabilityPlotter:
    """可靠性曲线绘制器
    
    绘制校准前后的可靠性图，这是评估校准效果的核心可视化。
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Args:
            figsize: 图片尺寸
            dpi: 图片分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 8)
    
    def plot_reliability_diagram(
        self,
        reliability_data: Dict[str, Any],
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
        show_bins: bool = True,
        show_histogram: bool = True
    ) -> plt.Figure:
        """绘制单个可靠性图
        
        Args:
            reliability_data: 包含bin数据的字典
            title: 图表标题
            save_path: 保存路径
            show_bins: 是否显示bin计数
            show_histogram: 是否显示置信度分布直方图
            
        Returns:
            matplotlib图表对象
        """
        if show_histogram:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
            ax_main = fig.add_subplot(gs[0, 0])
            ax_hist = fig.add_subplot(gs[1, 0])
        else:
            fig, ax_main = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 提取数据
        bin_centers = np.array(reliability_data['bin_centers'])
        bin_accuracies = np.array(reliability_data['bin_accuracies'])
        bin_confidences = np.array(reliability_data['bin_confidences'])
        bin_counts = np.array(reliability_data['bin_counts'])
        
        # 过滤空bin
        valid_bins = bin_counts > 0
        if valid_bins.sum() == 0:
            logger.warning("所有bin都为空，无法绘制可靠性图")
            return fig
        
        bin_centers = bin_centers[valid_bins]
        bin_accuracies = bin_accuracies[valid_bins]
        bin_confidences = bin_confidences[valid_bins]
        bin_counts = bin_counts[valid_bins]
        
        # 绘制理想对角线
        ax_main.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
        
        # 绘制可靠性曲线
        ax_main.plot(bin_confidences, bin_accuracies, 'o-', 
                    color=self.colors[0], linewidth=2, markersize=6,
                    label='Model Calibration')
        
        # 填充置信区间（基于bin大小）
        if show_bins:
            # 使用bin大小调整点的大小
            normalized_counts = bin_counts / bin_counts.max() * 200 + 50
            scatter = ax_main.scatter(bin_confidences, bin_accuracies, 
                                    s=normalized_counts, alpha=0.6, 
                                    color=self.colors[0], edgecolors='white', linewidth=1)
            
            # 添加bin计数标签
            for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
                if count > 0:
                    ax_main.annotate(f'{int(count)}', 
                                   (conf, acc), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, alpha=0.8)
        
        # 设置主图样式
        ax_main.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax_main.set_ylabel('Fraction of Positives', fontsize=12)
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        ax_main.legend(fontsize=11)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        
        # 添加ECE信息
        if 'ECE' in reliability_data:
            ece_text = f'ECE = {reliability_data["ECE"]:.4f}'
            ax_main.text(0.05, 0.95, ece_text, transform=ax_main.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=12, fontweight='bold')
        
        # 绘制置信度分布直方图
        if show_histogram and 'confidences' in reliability_data:
            confidences = reliability_data['confidences']
            ax_hist.hist(confidences, bins=30, density=True, alpha=0.7, 
                        color=self.colors[1], edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel('Confidence', fontsize=10)
            ax_hist.set_ylabel('Density', fontsize=10)
            ax_hist.set_title('Confidence Distribution', fontsize=11)
            ax_hist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"可靠性图已保存: {save_path}")
        
        return fig
    
    def plot_calibration_comparison(
        self,
        calibration_results: Dict[str, Dict],
        methods_order: Optional[List[str]] = None,
        title: str = "Calibration Methods Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """比较多种校准方法的可靠性图
        
        Args:
            calibration_results: 不同方法的校准结果
            methods_order: 方法显示顺序
            title: 图表标题
            save_path: 保存路径
        """
        n_methods = len(calibration_results)
        
        if n_methods <= 2:
            fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6), dpi=self.dpi)
            if n_methods == 1:
                axes = [axes]
        else:
            n_cols = min(3, n_methods)
            n_rows = (n_methods + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), dpi=self.dpi)
            axes = axes.flatten() if n_methods > 1 else [axes]
        
        # 确定方法顺序
        if methods_order is None:
            methods_order = list(calibration_results.keys())
        
        # 为每种方法绘制可靠性图
        for i, method_name in enumerate(methods_order):
            if i >= len(axes):
                break
                
            if method_name not in calibration_results:
                logger.warning(f"方法 {method_name} 不在结果中")
                continue
            
            data = calibration_results[method_name]
            ax = axes[i]
            
            # 绘制理想对角线
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect')
            
            # 提取并绘制数据
            if 'reliability_data' in data:
                rel_data = data['reliability_data']
                bin_centers = np.array(rel_data['bin_centers'])
                bin_accuracies = np.array(rel_data['bin_accuracies'])
                bin_confidences = np.array(rel_data['bin_confidences'])
                bin_counts = np.array(rel_data['bin_counts'])
                
                # 过滤有效bin
                valid_bins = bin_counts > 0
                if valid_bins.sum() > 0:
                    ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                           'o-', color=self.colors[i % len(self.colors)], 
                           linewidth=2, markersize=6, label=method_name)
            
            # 设置图表样式
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{method_name}\nECE = {data.get("ECE", 0):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 隐藏多余的子图
        for i in range(len(methods_order), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"校准对比图已保存: {save_path}")
        
        return fig


class PerformancePlotter:
    """性能对比图表绘制器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("Set2", 10)
    
    def plot_performance_vs_calibration(
        self,
        results_df: pd.DataFrame,
        x_metric: str = 'ECE',
        y_metric: str = 'macro_f1',
        title: str = "Performance vs Calibration",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制性能vs校准散点图"""
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制散点图
        scatter = ax.scatter(results_df[x_metric], results_df[y_metric], 
                           c=range(len(results_df)), cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # 添加方法标签
        if 'method' in results_df.columns:
            for i, (idx, row) in enumerate(results_df.iterrows()):
                ax.annotate(row['method'], 
                           (row[x_metric], row[y_metric]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.9)
        
        # 设置图表样式
        ax.set_xlabel(f'{x_metric.upper()}', fontsize=12)
        ax.set_ylabel(f'{y_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加目标线（如果适用）
        if x_metric == 'ECE':
            ax.axvline(x=0.08, color='red', linestyle='--', alpha=0.7, 
                      label='ECE Target (0.08)')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"性能散点图已保存: {save_path}")
        
        return fig
    
    def plot_metric_comparison_bar(
        self,
        results_df: pd.DataFrame,
        metrics: List[str],
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制指标对比条形图"""
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6), dpi=self.dpi)
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric in results_df.columns:
                # 按指标值排序
                sorted_df = results_df.sort_values(metric, ascending=False)
                
                bars = ax.bar(range(len(sorted_df)), sorted_df[metric], 
                             color=self.colors[i % len(self.colors)], alpha=0.7)
                
                # 添加数值标签
                for j, (idx, row) in enumerate(sorted_df.iterrows()):
                    ax.text(j, row[metric] + 0.001, f'{row[metric]:.3f}',
                           ha='center', va='bottom', fontsize=9)
                
                # 设置x轴标签
                ax.set_xticks(range(len(sorted_df)))
                ax.set_xticklabels(sorted_df['method'] if 'method' in sorted_df.columns else sorted_df.index,
                                  rotation=45, ha='right')
                
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.upper()} Comparison')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, f'Metric {metric} not found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"指标对比图已保存: {save_path}")
        
        return fig
    
    def plot_confusion_matrices(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        class_names: List[str],
        title: str = "Confusion Matrices Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制混淆矩阵对比图"""
        
        n_methods = len(confusion_matrices)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=self.dpi)
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, cm) in enumerate(confusion_matrices.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # 归一化混淆矩阵
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 绘制热力图
            im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            
            # 添加数值标签
            thresh = cm_norm.max() / 2.0
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(col, row, f'{cm[row, col]}\n({cm_norm[row, col]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_norm[row, col] > thresh else "black",
                           fontsize=9)
            
            ax.set_title(method_name)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        # 隐藏多余的子图
        for i in range(len(confusion_matrices), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"混淆矩阵对比图已保存: {save_path}")
        
        return fig
    
    def plot_routing_performance_curves(
        self,
        threshold_results: pd.DataFrame,
        title: str = "Routing Performance vs Threshold",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制路由性能曲线"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric in threshold_results.columns:
                ax.plot(threshold_results['threshold'], threshold_results[metric],
                       'o-', color=self.colors[i], linewidth=2, markersize=6)
                
                # 添加覆盖率信息（次要y轴）
                ax2 = ax.twinx()
                ax2.plot(threshold_results['threshold'], threshold_results['coverage'],
                        's--', color='gray', alpha=0.7, label='Coverage')
                ax2.set_ylabel('Coverage', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
            
            ax.set_xlabel('Confidence Threshold')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Threshold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 1.0)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"路由性能曲线已保存: {save_path}")
        
        return fig


class CalibrationVisualizer:
    """校准效果可视化器
    
    专门用于校准相关的可视化，包括校准前后对比、
    温度参数分析等。
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("Set1", 8)
    
    def plot_temperature_analysis(
        self,
        temperature_history: List[Dict],
        title: str = "Temperature Scaling Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制温度参数分析图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 提取温度和ECE数据
        temperatures = [entry['temperature'] for entry in temperature_history]
        eces = [entry['score'] for entry in temperature_history]
        
        # 左图：温度vs ECE曲线
        ax1.plot(temperatures, eces, 'o-', color=self.colors[0], linewidth=2)
        
        # 标记最优温度
        best_idx = np.argmin(eces)
        best_temp = temperatures[best_idx]
        best_ece = eces[best_idx]
        
        ax1.scatter([best_temp], [best_ece], color='red', s=100, zorder=5,
                   label=f'Optimal T={best_temp:.3f}')
        ax1.axvline(x=best_temp, color='red', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('ECE')
        ax1.set_title('Temperature vs ECE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：温度分布直方图
        ax2.hist(temperatures, bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(x=best_temp, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal T={best_temp:.3f}')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Temperature Search Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"温度分析图已保存: {save_path}")
        
        return fig
    
    def plot_calibration_improvement(
        self,
        before_after_results: Dict[str, Dict],
        title: str = "Calibration Improvement",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制校准改进效果图"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        methods = list(before_after_results.keys())
        metrics = ['ECE', 'MCE', 'brier_score', 'accuracy']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            before_values = []
            after_values = []
            method_names = []
            
            for method in methods:
                if 'before' in before_after_results[method] and 'after' in before_after_results[method]:
                    before_val = before_after_results[method]['before'].get(metric, 0)
                    after_val = before_after_results[method]['after'].get(metric, 0)
                    
                    before_values.append(before_val)
                    after_values.append(after_val)
                    method_names.append(method)
            
            if before_values and after_values:
                x = np.arange(len(method_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, before_values, width, label='Before', 
                              color=self.colors[0], alpha=0.7)
                bars2 = ax.bar(x + width/2, after_values, width, label='After',
                              color=self.colors[1], alpha=0.7)
                
                # 添加改进箭头
                for j, (before, after) in enumerate(zip(before_values, after_values)):
                    if metric in ['ECE', 'MCE', 'brier_score']:  # 越小越好
                        improvement = (before - after) / before * 100 if before > 0 else 0
                        arrow_color = 'green' if improvement > 0 else 'red'
                    else:  # accuracy越大越好
                        improvement = (after - before) / before * 100 if before > 0 else 0
                        arrow_color = 'green' if improvement > 0 else 'red'
                    
                    ax.annotate(f'{improvement:+.1f}%', 
                               xy=(j, max(before, after)), xytext=(j, max(before, after) * 1.1),
                               ha='center', va='bottom', color=arrow_color, fontweight='bold')
                
                ax.set_xlabel('Methods')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.upper()} Improvement')
                ax.set_xticks(x)
                ax.set_xticklabels(method_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"校准改进图已保存: {save_path}")
        
        return fig
    
    def create_comprehensive_report_figure(
        self,
        all_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """创建综合报告图表"""
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. 可靠性图对比 (左上, 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        if 'reliability_data' in all_results:
            # 这里应该调用可靠性图绘制逻辑
            ax1.set_title('Reliability Diagrams Comparison', fontsize=14, fontweight='bold')
        
        # 2. 指标对比条形图 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'metrics_comparison' in all_results:
            # 绘制关键指标对比
            ax2.set_title('Key Metrics', fontsize=12, fontweight='bold')
        
        # 3. 校准改进 (右中)
        ax3 = fig.add_subplot(gs[1, 2])
        if 'calibration_improvement' in all_results:
            # 绘制ECE改进
            ax3.set_title('ECE Improvement', fontsize=12, fontweight='bold')
        
        # 4. 混淆矩阵 (左下)
        ax4 = fig.add_subplot(gs[2, 0])
        if 'confusion_matrix' in all_results:
            # 绘制最佳方法的混淆矩阵
            ax4.set_title('Best Method CM', fontsize=12, fontweight='bold')
        
        # 5. 路由性能 (中下)
        ax5 = fig.add_subplot(gs[2, 1])
        if 'routing_performance' in all_results:
            # 绘制路由准确率
            ax5.set_title('Routing Accuracy', fontsize=12, fontweight='bold')
        
        # 6. 统计显著性 (右下)
        ax6 = fig.add_subplot(gs[2, 2])
        if 'statistical_tests' in all_results:
            # 绘制p值和效应大小
            ax6.set_title('Statistical Tests', fontsize=12, fontweight='bold')
        
        plt.suptitle('Experiment 1: Complexity Classifier Validation - Comprehensive Results', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"综合报告图已保存: {save_path}")
        
        return fig