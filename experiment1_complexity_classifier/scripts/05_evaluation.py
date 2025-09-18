#!/usr/bin/env python3
"""
实验一：综合评估脚本

功能:
1. 加载所有训练好的模型和校准器
2. 在测试集上进行全面评估
3. 计算所有核心指标
4. 生成对比图表和统计报告

运行方式:
    python scripts/05_evaluation.py
"""

import os
import sys
import yaml
import json
import pickle
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root))

from src.utils.metrics import ClassificationMetrics, CalibrationMetrics, StatisticalTests
from src.utils.visualization import PerformancePlotter, ReliabilityPlotter

# 配置日志到实验输出目录
experiment_root = Path(__file__).parent.parent
logs_dir = experiment_root / 'outputs' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / 'evaluation.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_test_data(data_dir: Path) -> tuple:
    """加载测试数据 - 支持JSON格式"""
    logger.info("加载测试数据...")
    
    # 加载JSON格式数据
    with open(data_dir / 'test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    X_test = [item['query'] for item in test_data]
    y_test = np.array([{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in test_data])
    
    logger.info(f"测试集: {len(X_test)} 条")
    
    return X_test, y_test


def load_baseline_results(results_dir: Path) -> dict:
    """加载基线模型结果"""
    logger.info("加载基线模型结果...")
    
    baseline_results = {}
    baseline_models = ['random', 'rule_based', 'bert', 'roberta']
    
    for model_name in baseline_models:
        result_file = results_dir / f"{model_name}_results.json"
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            baseline_results[model_name] = results
            logger.info(f"加载 {model_name} 结果")
        else:
            logger.warning(f"未找到 {model_name} 结果文件")
    
    return baseline_results


def load_modernbert_results(results_dir: Path) -> dict:
    """加载ModernBERT结果"""
    logger.info("加载ModernBERT结果...")
    
    modernbert_results = {}
    
    # 加载未校准的ModernBERT结果
    final_results_file = results_dir / 'modernbert_final_results.json'
    if final_results_file.exists():
        with open(final_results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        modernbert_results['modernbert_uncalibrated'] = results
        logger.info("加载ModernBERT未校准结果")
    
    return modernbert_results


def load_calibration_results(results_dir: Path) -> dict:
    """加载校准结果"""
    logger.info("加载校准结果...")
    
    calibration_file = results_dir / 'calibration_results.json'
    
    if calibration_file.exists():
        with open(calibration_file, 'r', encoding='utf-8') as f:
            calibration_results = json.load(f)
        logger.info("加载校准结果")
        return calibration_results
    else:
        logger.warning("未找到校准结果文件")
        return {}


def aggregate_all_results(baseline_results, modernbert_results, calibration_results, y_test):
    """聚合所有结果"""
    logger.info("聚合所有模型结果...")
    
    aggregated_results = {}
    
    # 处理基线模型结果
    for model_name, results in baseline_results.items():
        if 'classification' in results and 'calibration' in results:
            aggregated_results[model_name] = {
                'model_type': 'baseline',
                'accuracy': results['classification']['accuracy'],
                'macro_f1': results['classification']['macro_f1'],
                'micro_f1': results['classification']['micro_f1'],
                'ECE': results['calibration']['ECE'],
                'MCE': results['calibration']['MCE'],
                'brier_score': results['calibration']['brier_score'],
                'nll': results['calibration']['nll']
            }
    
    # 处理ModernBERT未校准结果
    for model_name, results in modernbert_results.items():
        if 'classification' in results and 'calibration' in results:
            aggregated_results[model_name] = {
                'model_type': 'modernbert',
                'accuracy': results['classification']['accuracy'],
                'macro_f1': results['classification']['macro_f1'],
                'micro_f1': results['classification']['micro_f1'],
                'ECE': results['calibration']['ECE'],
                'MCE': results['calibration']['MCE'],
                'brier_score': results['calibration']['brier_score'],
                'nll': results['calibration']['nll']
            }
    
    # 处理校准结果（不再用accuracy近似F1，缺失则置为NaN）
    for method_name, results in calibration_results.items():
        if results is not None and isinstance(results, dict):
            model_key = f"modernbert_{method_name}"
            aggregated_results[model_key] = {
                'model_type': 'calibrated',
                'calibration_method': method_name,
                'accuracy': results.get('accuracy', np.nan),
                'macro_f1': np.nan,
                'micro_f1': np.nan,
                'ECE': results.get('ECE', np.nan),
                'MCE': results.get('MCE', np.nan),
                'brier_score': results.get('brier_score', np.nan),
                'nll': results.get('nll', np.nan)
            }
    
    # 若无任何真实结果，则直接报错并退出，禁止使用模拟数据
    if not aggregated_results:
        logger.error("未找到任何真实评估结果。请先运行基线、ModernBERT训练与校准脚本以生成真实结果文件。")
        return {}
    
    return aggregated_results


    # 删除模拟评估结果函数


def perform_statistical_tests(aggregated_results):
    """执行统计显著性检验（仅在提供多次运行结果时进行）。"""
    logger.info("执行统计显著性检验...")
    
    # 目前聚合结果为单次评估，缺少重复实验样本，谨慎起见不进行基于模拟的统计检验。
    # 如需统计检验，请提供每个模型的多次运行指标数组（例如交叉验证/Bootstrap）。
    return {}


def create_comprehensive_evaluation_report(aggregated_results, statistical_tests, output_dir):
    """创建综合评估报告"""
    logger.info("创建综合评估报告...")
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建主要结果表格
    main_results = []
    for model_name, results in aggregated_results.items():
        row = {
            'Model': model_name,
            'Type': results['model_type'],
            'Accuracy': results['accuracy'],
            'Macro-F1': results['macro_f1'],
            'Micro-F1': results['micro_f1'],
            'ECE': results['ECE'],
            'MCE': results['MCE'],
            'Brier Score': results['brier_score']
        }
        
        if results['model_type'] == 'calibrated':
            row['Calibration Method'] = results.get('calibration_method', 'N/A')
        
        main_results.append(row)
    
    main_df = pd.DataFrame(main_results)
    main_df = main_df.round(4)
    
    # 按不同标准排序
    accuracy_ranking = main_df.sort_values('Accuracy', ascending=False)
    f1_ranking = main_df.sort_values('Macro-F1', ascending=False)
    ece_ranking = main_df.sort_values('ECE', ascending=True)  # ECE越低越好
    
    # 保存主要结果表格
    main_table_file = results_dir / 'comprehensive_evaluation_results.csv'
    main_df.to_csv(main_table_file, index=False)
    logger.info(f"保存主要结果表格: {main_table_file}")
    
    # 生成详细报告
    report_file = results_dir / 'comprehensive_evaluation_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 实验一：复杂度分类器综合评估报告\\n\\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## 实验目标与成功标准\\n\\n")
        f.write("### 核心验证指标 ⭐\\n")
        f.write("- **★★★ ECE < 0.08** (温度缩放校准效果)\\n")
        f.write("- **★★★ 分类准确率 > 85%** (ModernBERT有效性)\\n")
        f.write("- **★★ 路由准确率 > 90%** (系统可用性)\\n\\n")
        
        f.write("## 整体性能对比\\n\\n")
        f.write(main_df.to_markdown(index=False))
        f.write("\\n\\n")
        
        f.write("## 关键发现\\n\\n")
        
        # 最佳性能模型
        best_accuracy = accuracy_ranking.iloc[0]
        best_f1 = f1_ranking.iloc[0]
        best_ece = ece_ranking.iloc[0]
        
        f.write(f"### 性能冠军\\n")
        f.write(f"- **最高准确率**: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})\\n")
        f.write(f"- **最高Macro-F1**: {best_f1['Model']} ({best_f1['Macro-F1']:.4f})\\n")
        f.write(f"- **最佳校准(最低ECE)**: {best_ece['Model']} ({best_ece['ECE']:.4f})\\n\\n")
        
        # 目标达成情况
        f.write("### 目标达成情况\\n")
        
        accuracy_achievers = main_df[main_df['Accuracy'] >= 0.85]
        ece_achievers = main_df[main_df['ECE'] <= 0.08]
        
        f.write(f"- **准确率达标(≥85%)**: {len(accuracy_achievers)} 个模型\\n")
        for _, model in accuracy_achievers.iterrows():
            f.write(f"  - {model['Model']}: {model['Accuracy']:.4f}\\n")
        
        f.write(f"- **ECE达标(≤0.08)**: {len(ece_achievers)} 个模型\\n")
        for _, model in ece_achievers.iterrows():
            f.write(f"  - {model['Model']}: {model['ECE']:.4f}\\n")
        
        f.write("\\n")
        
        # 校准效果分析
        f.write("### 校准效果分析\\n")
        
        calibrated_models = main_df[main_df['Type'] == 'calibrated']
        if not calibrated_models.empty:
            uncalibrated_ece = main_df[main_df['Model'].str.contains('uncalibrated')]['ECE'].iloc[0] if len(main_df[main_df['Model'].str.contains('uncalibrated')]) > 0 else 0.15
            
            f.write(f"未校准ECE基线: {uncalibrated_ece:.4f}\\n\\n")
            
            for _, model in calibrated_models.iterrows():
                if 'uncalibrated' not in model['Model']:
                    improvement = (uncalibrated_ece - model['ECE']) / uncalibrated_ece * 100
                    f.write(f"- **{model.get('Calibration Method', 'Unknown')}**: ")
                    f.write(f"ECE={model['ECE']:.4f} (改进{improvement:.1f}%)\\n")
        
        f.write("\\n")
        
        # 统计显著性结果
        if statistical_tests:
            f.write("### 统计显著性检验\\n")
            f.write("| 模型 | p-value | 显著性 | Cohen's d | 效应大小 | 95% CI |\\n")
            f.write("|------|---------|--------|-----------|----------|--------|\\n")
            
            for model_name, test_result in statistical_tests.items():
                f.write(f"| {model_name} | {test_result['t_test_p_value']:.4f} | ")
                f.write(f"{'✓' if test_result['t_test_significant'] else '✗'} | ")
                f.write(f"{test_result['cohens_d']:.3f} | {test_result['effect_size']} | ")
                f.write(f"[{test_result['ci_lower']:.3f}, {test_result['ci_upper']:.3f}] |\\n")
        
        f.write("\\n")
        
        f.write("## 模型分类分析\\n\\n")
        
        # 基线模型分析
        baseline_models = main_df[main_df['Type'] == 'baseline']
        if not baseline_models.empty:
            f.write("### 基线模型表现\\n")
            baseline_sorted = baseline_models.sort_values('Macro-F1', ascending=False)
            for _, model in baseline_sorted.iterrows():
                f.write(f"- **{model['Model']}**: ")
                f.write(f"F1={model['Macro-F1']:.4f}, ECE={model['ECE']:.4f}\\n")
            f.write("\\n")
        
        # ModernBERT分析
        modernbert_models = main_df[main_df['Type'].isin(['modernbert', 'calibrated'])]
        if not modernbert_models.empty:
            f.write("### ModernBERT及校准方法表现\\n")
            modernbert_sorted = modernbert_models.sort_values('ECE', ascending=True)
            for _, model in modernbert_sorted.iterrows():
                f.write(f"- **{model['Model']}**: ")
                f.write(f"准确率={model['Accuracy']:.4f}, ECE={model['ECE']:.4f}\\n")
            f.write("\\n")
        
        f.write("## 结论与建议\\n\\n")
        
        # 基于结果的结论
        if len(ece_achievers) > 0:
            f.write("✓ **校准目标达成**: 成功实现ECE < 0.08的校准效果\\n")
        else:
            f.write("✗ **校准目标未完全达成**: 需要进一步优化校准方法\\n")
        
        if len(accuracy_achievers) > 0:
            f.write("✓ **准确率目标达成**: ModernBERT展现出良好的分类性能\\n")
        else:
            f.write("✗ **准确率目标未达成**: 需要改进模型训练策略\\n")
        
        f.write("\\n")
        
        # 推荐的最佳模型
        f.write("### 推荐模型\\n")
        
        # 综合评分：准确率权重0.4，ECE改进权重0.6
        main_df['composite_score'] = main_df['Accuracy'] * 0.4 + (0.2 - main_df['ECE']) * 0.6
        best_overall = main_df.loc[main_df['composite_score'].idxmax()]
        
        f.write(f"**推荐使用**: {best_overall['Model']}\\n")
        f.write(f"- 准确率: {best_overall['Accuracy']:.4f}\\n")
        f.write(f"- ECE: {best_overall['ECE']:.4f}\\n")
        f.write(f"- 综合评分: {best_overall['composite_score']:.4f}\\n")
    
    logger.info(f"保存综合评估报告: {report_file}")
    
    return main_df


def create_evaluation_visualizations(aggregated_results, output_dir):
    """创建评估可视化"""
    logger.info("创建评估可视化...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 转换为DataFrame用于可视化
        plot_data = []
        for model_name, results in aggregated_results.items():
            plot_data.append({
                'method': model_name,
                'accuracy': results['accuracy'],
                'macro_f1': results['macro_f1'],
                'ECE': results['ECE'],
                'MCE': results['MCE'],
                'brier_score': results['brier_score']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        plotter = PerformancePlotter()
        
        # 1. 性能vs校准散点图
        fig1 = plotter.plot_performance_vs_calibration(
            plot_df,
            x_metric='ECE',
            y_metric='macro_f1',
            title='Model Performance vs Calibration Quality'
        )
        
        scatter_file = figures_dir / 'performance_vs_calibration_scatter.png'
        fig1.savefig(scatter_file, dpi=300, bbox_inches='tight')
        logger.info(f"保存性能散点图: {scatter_file}")
        
        # 2. 核心指标对比条形图
        fig2 = plotter.plot_metric_comparison_bar(
            plot_df,
            metrics=['accuracy', 'macro_f1', 'ECE'],
            title='Core Metrics Comparison Across All Models'
        )
        
        bar_file = figures_dir / 'core_metrics_comparison.png'
        fig2.savefig(bar_file, dpi=300, bbox_inches='tight')
        logger.info(f"保存指标对比图: {bar_file}")
        
        # 3. 校准方法专门对比
        calibrated_methods = plot_df[plot_df['method'].str.contains('modernbert')]
        if not calibrated_methods.empty:
            fig3 = plotter.plot_metric_comparison_bar(
                calibrated_methods,
                metrics=['ECE', 'MCE', 'brier_score'],
                title='Calibration Methods Comparison (ModernBERT)'
            )
            
            calibration_file = figures_dir / 'calibration_methods_comparison.png'
            fig3.savefig(calibration_file, dpi=300, bbox_inches='tight')
            logger.info(f"保存校准对比图: {calibration_file}")
        
    except Exception as e:
        logger.warning(f"创建可视化失败: {e}")


def save_evaluation_artifacts(aggregated_results, statistical_tests, main_df, output_dir):
    """保存评估文件"""
    logger.info("保存评估文件...")
    
    results_dir = output_dir / 'results'
    
    # 保存聚合结果
    aggregated_file = results_dir / 'aggregated_evaluation_results.json'
    with open(aggregated_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
    logger.info(f"保存聚合结果: {aggregated_file}")
    
    # 保存统计检验结果
    if statistical_tests:
        stats_file = results_dir / 'statistical_tests_results.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistical_tests, f, indent=2, ensure_ascii=False)
        logger.info(f"保存统计检验结果: {stats_file}")
    
    # 保存排名信息
    rankings = {
        'by_accuracy': main_df.sort_values('Accuracy', ascending=False)[['Model', 'Accuracy']].to_dict('records'),
        'by_macro_f1': main_df.sort_values('Macro-F1', ascending=False)[['Model', 'Macro-F1']].to_dict('records'),
        'by_ece': main_df.sort_values('ECE', ascending=True)[['Model', 'ECE']].to_dict('records'),
    }
    
    rankings_file = results_dir / 'model_rankings.json'
    with open(rankings_file, 'w', encoding='utf-8') as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)
    logger.info(f"保存排名信息: {rankings_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合评估")
    parser.add_argument('--config', type=str, default='config/evaluation_config.yaml',
                       help='评估配置文件路径')
    # 移除模拟数据开关：评估仅基于真实结果
    args = parser.parse_args()
    
    logger.info("开始综合评估...")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载测试数据
    data_dir = Path('data/processed')
    X_test, y_test = load_test_data(data_dir)
    
    # 加载各种结果
    results_dir = Path('outputs/results')
    
    baseline_results = load_baseline_results(results_dir)
    modernbert_results = load_modernbert_results(results_dir)
    calibration_results = load_calibration_results(results_dir)
    
    # 聚合所有结果
    aggregated_results = aggregate_all_results(
        baseline_results, modernbert_results, calibration_results, y_test
    )
    
    # 执行统计显著性检验
    statistical_tests = perform_statistical_tests(aggregated_results)
    
    # 创建综合评估报告
    output_dir = Path('outputs')
    main_df = create_comprehensive_evaluation_report(
        aggregated_results, statistical_tests, output_dir
    )
    
    # 创建可视化
    create_evaluation_visualizations(aggregated_results, output_dir)
    
    # 保存评估文件
    save_evaluation_artifacts(aggregated_results, statistical_tests, main_df, output_dir)
    
    # 输出关键结果
    logger.info("\\n=== 综合评估完成 ===")
    
    # 目标达成检查
    accuracy_achievers = main_df[main_df['Accuracy'] >= 0.85]
    ece_achievers = main_df[main_df['ECE'] <= 0.08]
    
    logger.info(f"准确率达标模型: {len(accuracy_achievers)} 个")
    logger.info(f"ECE达标模型: {len(ece_achievers)} 个")
    
    if len(accuracy_achievers) > 0 and len(ece_achievers) > 0:
        logger.info("✓ 实验目标达成！")
    else:
        logger.warning("✗ 实验目标部分未达成")
    
    # 推荐模型
    best_accuracy = main_df.loc[main_df['Accuracy'].idxmax()]
    best_ece = main_df.loc[main_df['ECE'].idxmin()]
    
    logger.info("\\n推荐模型:")
    logger.info(f"最高准确率: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    logger.info(f"最佳校准: {best_ece['Model']} (ECE={best_ece['ECE']:.4f})")
    
    logger.info("综合评估完毕!")


if __name__ == "__main__":
    main()