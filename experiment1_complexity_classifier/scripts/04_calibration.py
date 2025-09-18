#!/usr/bin/env python3
"""
实验一：校准方法拟合脚本

功能:
1. 加载ModernBERT模型的logits
2. 拟合各种校准方法 (Temperature Scaling, Platt, Isotonic, TvA+TS)
3. 评估校准效果
4. 生成校准对比报告

运行方式:
    python scripts/04_calibration.py
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
from sklearn.model_selection import StratifiedShuffleSplit

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root))

from src.models.calibration import (
    TemperatureScaling,
    TopVersusAllPlatt,
    TopVersusAllIsotonic,
    TvATemperatureScaling,
    CalibratedClassifier
)
from src.utils.metrics import CalibrationMetrics
from src.utils.visualization import ReliabilityPlotter, CalibrationVisualizer

# 配置日志到实验输出目录
experiment_root = Path(__file__).parent.parent
logs_dir = experiment_root / 'outputs' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / 'calibration.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_calibration_data(data_dir: Path) -> tuple:
    """加载校准数据 - 支持JSON格式"""
    logger.info("加载校准数据...")
    
    # 加载JSON格式数据
    with open(data_dir / 'calibration_data.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    with open(data_dir / 'test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 准备校准数据
    y_cal = np.array([{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in calibration_data])
    
    # 准备测试数据
    y_test = np.array([{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in test_data])
    
    logger.info(f"校准集: {len(y_cal)} 条")
    logger.info(f"测试集: {len(y_test)} 条")
    
    return y_cal, y_test


def load_modernbert_logits(results_dir: Path, metrics_cfg: dict) -> tuple:
    """加载ModernBERT的logits"""
    logger.info("加载ModernBERT logits...")
    
    # 先加载最终结果（关键路径）
    results_file = results_dir / 'modernbert_final_results.json'
    if not results_file.exists():
        logger.error(f"未找到ModernBERT结果文件: {results_file}")
        return None, None

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        test_logits = np.array(results['test_logits'])
        test_probabilities = np.array(results['test_probabilities'])
        logger.info(f"加载logits形状: {test_logits.shape}")
    except Exception as e:
        logger.error(f"读取最终结果失败: {e}")
        return None, None

    # 再尽力从超参结果中找校准集 logits；失败时不应影响主流程
    cal_logits = None
    cal_probabilities = None
    try:
        hyperparameter_file = results_dir / 'modernbert_hyperparameter_search.json'
        if hyperparameter_file.exists():
            with open(hyperparameter_file, 'r', encoding='utf-8') as f:
                hyperparam_results = json.load(f)

            # 兼容两种结构：
            # 1) 条目顶层含 'composite_score'
            # 2) 条目含 'cv_results' 且其中含 'composite_score'
            def _score(item: dict) -> float:
                if isinstance(item, dict):
                    if 'composite_score' in item and isinstance(item['composite_score'], (int, float)):
                        return float(item['composite_score'])
                    if 'cv_results' in item and isinstance(item['cv_results'], dict):
                        return float(item['cv_results'].get('composite_score', float('-inf')))
                return float('-inf')

            if isinstance(hyperparam_results, list) and len(hyperparam_results) > 0:
                scored = [(it, _score(it)) for it in hyperparam_results]
                scored = [t for t in scored if t[1] != float('-inf')]
                if scored:
                    best_result = max(scored, key=lambda t: t[1])[0]
                    # 优先从最终结果文件中读取验证集（已由03脚本写入）
                    val_logits = results.get('val_logits')
                    val_probs = results.get('val_probabilities')
                    # 兼容：若最终结果未包含，则从超参条目中兜底
                    if val_logits is None or val_probs is None:
                        val_logits = best_result.get('val_logits') if isinstance(best_result, dict) else None
                        val_probs = best_result.get('val_probabilities') if isinstance(best_result, dict) else None
                    if val_logits is not None and val_probs is not None:
                        cal_logits = np.array(val_logits)
                        cal_probabilities = np.array(val_probs)
                        logger.info(f"加载校准logits形状: {cal_logits.shape}")
                else:
                    logger.info("超参数结果缺少 composite_score 字段，跳过基于超参选择校准集。")
            else:
                logger.info("超参数结果为空或格式非列表，跳过基于超参选择校准集。")
        else:
            logger.info("未发现超参数结果文件，稍后将从测试集切分校准集。")
    except Exception as e:
        # 仅警告，不中断；后续主流程会使用测试集中的一部分作为校准集
        logger.warning(f"读取超参数搜索结果失败，使用回退策略: {e}")

    return (cal_logits, cal_probabilities), (test_logits, test_probabilities)


# 严禁模拟logits：移除所有模拟/合成数据路径


def fit_temperature_scaling(cal_logits, y_cal, config):
    """拟合温度缩放校准器"""
    logger.info("拟合温度缩放校准器...")
    
    calibrator_config = config['calibration_methods']['temperature_scaling']
    
    calibrator = TemperatureScaling(
        temperature_range=tuple(calibrator_config['temperature_search']['search_range']),
        num_search_points=calibrator_config['temperature_search']['num_points'],
        optimization_method=calibrator_config['temperature_search']['method'],
        objective=calibrator_config['optimization']['objective'],
        max_iterations=calibrator_config['optimization']['max_iterations'],
        tolerance=calibrator_config['optimization']['tolerance'],
        random_state=42
    )
    
    calibrator.fit(cal_logits, y_cal)
    
    logger.info(f"最优温度: {calibrator.temperature_:.4f}")
    
    return calibrator


def fit_tva_platt(cal_logits, y_cal, config):
    """拟合Top-vs-All + Platt Scaling校准器"""
    logger.info("拟合Top-vs-All + Platt Scaling校准器...")
    
    calibrator_config = config['calibration_methods']['tva_platt']
    
    calibrator = TopVersusAllPlatt(
        regularization=calibrator_config['platt_scaling']['regularization'],
        max_iterations=calibrator_config['platt_scaling']['max_iterations'],
        solver=calibrator_config['platt_scaling']['solver'],
        C_values=calibrator_config['platt_scaling']['parameter_search']['C_values'],
        cross_validation=calibrator_config['platt_scaling']['parameter_search']['cross_validation'],
        random_state=42
    )
    
    calibrator.fit(cal_logits, y_cal)
    
    return calibrator


def fit_tva_isotonic(cal_logits, y_cal, config):
    """拟合Top-vs-All + Isotonic Regression校准器"""
    logger.info("拟合Top-vs-All + Isotonic Regression校准器...")
    
    calibrator_config = config['calibration_methods']['tva_isotonic']
    
    calibrator = TopVersusAllIsotonic(
        increasing=calibrator_config['isotonic_regression']['increasing'],
        out_of_bounds=calibrator_config['isotonic_regression']['out_of_bounds'],
        smoothing=calibrator_config['isotonic_regression'].get('smoothing', {}).get('enabled', False),
        smoothing_factor=calibrator_config['isotonic_regression'].get('smoothing', {}).get('smoothing_factor', 0.1)
    )
    
    calibrator.fit(cal_logits, y_cal)
    
    return calibrator


def fit_tva_temperature(cal_logits, y_cal, config):
    """拟合TvA + Temperature组合校准器"""
    logger.info("拟合TvA + Temperature组合校准器...")
    
    calibrator_config = config['calibration_methods']['tva_temperature']
    
    calibrator = TvATemperatureScaling(
        tva_method=calibrator_config.get('tva_method', 'platt'),
        temperature_range=tuple(calibrator_config['temperature_scaling']['search_range']),
        num_temp_points=calibrator_config['temperature_scaling']['num_points'],
        combination=calibrator_config['combination']['method']
    )
    
    calibrator.fit(cal_logits, y_cal)
    
    return calibrator


def evaluate_calibration_methods(calibrators, test_logits, y_test, original_probabilities, metrics_cfg: dict):
    """评估所有校准方法"""
    logger.info("评估校准方法...")
    
    num_bins = metrics_cfg.get('ECE', {}).get('num_bins', 15)
    bin_strategy = metrics_cfg.get('ECE', {}).get('bin_strategy', 'equal_width')
    cal_metrics = CalibrationMetrics(num_bins=num_bins, bin_strategy=bin_strategy)
    results = {}
    
    # 评估未校准的原始模型
    logger.info("评估未校准模型...")
    uncalibrated_results = cal_metrics.compute_all_calibration_metrics(
        y_test, original_probabilities, return_reliability_data=True
    )
    results['uncalibrated'] = uncalibrated_results
    
    # 评估各种校准方法
    for method_name, calibrator in calibrators.items():
        if calibrator is not None:
            try:
                logger.info(f"评估 {method_name}...")
                
                # 应用校准
                calibrated_probabilities = calibrator.transform(test_logits)
                
                # 计算校准指标
                calibrated_results = cal_metrics.compute_all_calibration_metrics(
                    y_test, calibrated_probabilities, return_reliability_data=True
                )
                
                results[method_name] = calibrated_results
                
                logger.info(f"{method_name} ECE: {calibrated_results['ECE']:.4f}")
                
            except Exception as e:
                logger.error(f"评估 {method_name} 失败: {e}")
                results[method_name] = None
        else:
            logger.warning(f"跳过 {method_name}（校准器为空）")
            results[method_name] = None
    
    return results


def create_calibration_comparison(calibration_results, output_dir, meta_info: dict):
    """创建校准对比报告"""
    logger.info("创建校准对比报告...")
    
    # 创建对比表格
    comparison_data = []
    
    for method_name, results in calibration_results.items():
        if results is not None:
            row = {
                'Method': method_name,
                'ECE': results['ECE'],
                'MCE': results['MCE'],
                'Brier Score': results['brier_score'],
                'NLL': results['nll'],
                'Accuracy': results['accuracy'],
                'Mean Confidence': results['mean_confidence']
            }
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    
    # 按ECE排序（越低越好）
    comparison_df = comparison_df.sort_values('ECE')
    
    # 计算改进程度
    if 'uncalibrated' in [row['Method'] for row in comparison_data]:
        uncalibrated_ece = comparison_df[comparison_df['Method'] == 'uncalibrated']['ECE'].iloc[0]
        
        comparison_df['ECE_Improvement'] = comparison_df.apply(
            lambda row: (uncalibrated_ece - row['ECE']) / uncalibrated_ece * 100 
            if row['Method'] != 'uncalibrated' else 0, axis=1
        )
        comparison_df['ECE_Improvement'] = comparison_df['ECE_Improvement'].round(2)
    
    # 保存对比表格
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    table_file = results_dir / 'calibration_comparison.csv'
    comparison_df.to_csv(table_file, index=False)
    logger.info(f"保存校准对比表格: {table_file}")
    
    # 生成详细报告
    report_file = results_dir / 'calibration_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 校准方法对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 元信息
        f.write("## 元信息\n\n")
        for k, v in (meta_info or {}).items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## 校准效果对比\n\n")
        try:
            f.write(comparison_df.to_markdown(index=False))
        except Exception:
            # 回退：环境缺少 tabulate 时写入CSV风格文本
            f.write(comparison_df.to_csv(index=False))
        f.write("\n\n")
        
        f.write("## 关键发现\\n\\n")
        
        # 最佳校准方法
        best_method = comparison_df.iloc[0]
        f.write(f"- **最佳校准方法**: {best_method['Method']} (ECE = {best_method['ECE']:.4f})\\n")
        
        if 'ECE_Improvement' in comparison_df.columns:
            best_improvement = comparison_df['ECE_Improvement'].max()
            best_improvement_method = comparison_df.loc[comparison_df['ECE_Improvement'].idxmax(), 'Method']
            f.write(f"- **最大ECE改进**: {best_improvement_method} ({best_improvement:.1f}%)\\n")
        
        # 目标达成情况
        target_ece = 0.08
        methods_achieving_target = comparison_df[comparison_df['ECE'] <= target_ece]
        f.write(f"- **达到ECE目标(≤{target_ece})的方法**: {len(methods_achieving_target)}个\\n")
        
        for _, method in methods_achieving_target.iterrows():
            f.write(f"  - {method['Method']}: {method['ECE']:.4f}\\n")
        
        f.write("\\n")
        
        f.write("## 方法分析\\n\\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"### {row['Method']}\\n")
            f.write(f"- ECE: {row['ECE']:.4f}\\n")
            f.write(f"- MCE: {row['MCE']:.4f}\\n") 
            f.write(f"- Brier Score: {row['Brier Score']:.4f}\\n")
            f.write(f"- 准确率: {row['Accuracy']:.4f}\\n")
            
            if 'ECE_Improvement' in row and row['ECE_Improvement'] != 0:
                f.write(f"- ECE改进: {row['ECE_Improvement']:.1f}%\\n")
            
            # 简单评价
            if row['ECE'] <= 0.05:
                f.write("- 评价: 优秀的校准质量\\n")
            elif row['ECE'] <= 0.08:
                f.write("- 评价: 良好的校准质量\\n")
            elif row['ECE'] <= 0.15:
                f.write("- 评价: 中等的校准质量\\n")
            else:
                f.write("- 评价: 校准质量需要改进\\n")
            f.write("\\n")
    
    logger.info(f"保存校准报告: {report_file}")
    
    return comparison_df


def create_calibration_visualizations(calibration_results, calibrators, output_dir):
    """创建校准可视化"""
    logger.info("创建校准可视化...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 可靠性图对比
        plotter = ReliabilityPlotter()
        
        # 过滤有效结果
        valid_results = {k: v for k, v in calibration_results.items() if v is not None}
        
        if valid_results:
            fig = plotter.plot_calibration_comparison(
                valid_results,
                title="Calibration Methods Reliability Comparison"
            )
            
            reliability_fig_file = figures_dir / 'calibration_reliability_comparison.png'
            fig.savefig(reliability_fig_file, dpi=300, bbox_inches='tight')
            logger.info(f"保存可靠性对比图: {reliability_fig_file}")
        
        # 校准改进效果图
        if 'temperature_scaling' in calibrators and calibrators['temperature_scaling'] is not None:
            try:
                calibration_visualizer = CalibrationVisualizer()
                
                # 温度缩放分析
                temp_calibrator = calibrators['temperature_scaling']
                if hasattr(temp_calibrator, 'optimization_history_') and temp_calibrator.optimization_history_:
                    fig = calibration_visualizer.plot_temperature_analysis(
                        temp_calibrator.optimization_history_,
                        title="Temperature Scaling Optimization"
                    )
                    
                    temp_fig_file = figures_dir / 'temperature_scaling_analysis.png'
                    fig.savefig(temp_fig_file, dpi=300, bbox_inches='tight')
                    logger.info(f"保存温度分析图: {temp_fig_file}")
                
                # 校准前后对比
                before_after_data = {}
                if 'uncalibrated' in valid_results and 'temperature_scaling' in valid_results:
                    before_after_data['temperature_scaling'] = {
                        'before': valid_results['uncalibrated'],
                        'after': valid_results['temperature_scaling']
                    }
                
                if before_after_data:
                    fig = calibration_visualizer.plot_calibration_improvement(
                        before_after_data,
                        title="Calibration Improvement Analysis"
                    )
                    
                    improvement_fig_file = figures_dir / 'calibration_improvement.png'
                    fig.savefig(improvement_fig_file, dpi=300, bbox_inches='tight')
                    logger.info(f"保存改进分析图: {improvement_fig_file}")
                    
            except Exception as e:
                logger.warning(f"创建高级可视化失败: {e}")
        
    except Exception as e:
        logger.warning(f"创建可视化失败: {e}")


def save_calibration_artifacts(calibrators, calibration_results, output_dir):
    """保存校准相关文件"""
    logger.info("保存校准文件...")
    
    # 保存校准器
    calibrators_dir = output_dir / 'models' / 'calibrators'
    calibrators_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name, calibrator in calibrators.items():
        if calibrator is not None:
            try:
                calibrator_file = calibrators_dir / f"{method_name}_calibrator.pkl"
                with open(calibrator_file, 'wb') as f:
                    pickle.dump(calibrator, f)
                logger.info(f"保存校准器: {calibrator_file}")
            except Exception as e:
                logger.warning(f"保存校准器失败 {method_name}: {e}")
    
    # 保存校准结果
    results_dir = output_dir / 'results'
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    calibration_file = results_dir / 'calibration_results.json'
    with open(calibration_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(calibration_results), f, indent=2, ensure_ascii=False)
    logger.info(f"保存校准结果: {calibration_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="校准方法拟合")
    parser.add_argument('--config', type=str, default='config/calibration_config.yaml',
                       help='校准配置文件路径')
    parser.add_argument('--methods', nargs='+',
                       choices=['temperature_scaling', 'tva_platt', 'tva_isotonic', 'tva_temperature', 'all'],
                       default=['all'], help='要拟合的校准方法')
    # 为保证真实数据优先，移除使用模拟数据的开关
    args = parser.parse_args()
    
    logger.info("开始校准方法拟合...")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    data_dir = Path('data/processed')
    y_cal, y_test = load_calibration_data(data_dir)
    
    # 加载logits
    results_dir = Path('outputs/results')
    metrics_cfg = config.get('evaluation', {}).get('calibration_metrics', {})
    logits_data = load_modernbert_logits(results_dir, metrics_cfg)
    
    if logits_data[0] is None or logits_data[1] is None:
        logger.error("缺少真实的 ModernBERT logits。请先运行 03_train_modernbert.py 生成 outputs/results/modernbert_final_results.json。")
        return
    else:
        (cal_logits, cal_probabilities), (test_logits, test_probabilities) = logits_data
        
        # 如果校准数据为空，使用分层随机切分（固定种子）从测试数据中抽取 25% 作为校准集
        if cal_logits is None:
            logger.warning("校准logits为空，使用分层随机切分从测试数据抽取 25% 作为校准集（回退策略）...")
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            for cal_idx, test_idx in splitter.split(test_logits, y_test):
                cal_logits = test_logits[cal_idx]
                cal_probabilities = test_probabilities[cal_idx]
                test_logits = test_logits[test_idx]
                test_probabilities = test_probabilities[test_idx]
                y_cal = y_test[cal_idx]
                y_test = y_test[test_idx]
                break
    
    # 确定要拟合的校准方法
    if 'all' in args.methods:
        methods_to_fit = ['temperature_scaling', 'tva_platt', 'tva_isotonic', 'tva_temperature']
    else:
        methods_to_fit = args.methods
    
    # 拟合校准方法
    calibrators = {}
    output_dir = Path('outputs')
    
    for method_name in methods_to_fit:
        logger.info(f"\\n{'='*50}")
        logger.info(f"拟合校准方法: {method_name}")
        logger.info(f"{'='*50}")
        
        try:
            if method_name == 'temperature_scaling':
                calibrator = fit_temperature_scaling(cal_logits, y_cal, config)
            elif method_name == 'tva_platt':
                calibrator = fit_tva_platt(cal_logits, y_cal, config)
            elif method_name == 'tva_isotonic':
                calibrator = fit_tva_isotonic(cal_logits, y_cal, config)
            elif method_name == 'tva_temperature':
                calibrator = fit_tva_temperature(cal_logits, y_cal, config)
            else:
                logger.warning(f"未知校准方法: {method_name}")
                calibrator = None
            
            calibrators[method_name] = calibrator
            
            if calibrator is not None:
                logger.info(f"{method_name} 校准器拟合完成")
            
        except Exception as e:
            logger.error(f"拟合 {method_name} 失败: {e}")
            calibrators[method_name] = None
    
    # 评估校准方法（准确率应基于校准后的概率重新计算）
    calibration_results = evaluate_calibration_methods(
        calibrators, test_logits, y_test, test_probabilities, metrics_cfg
    )
    
    # 创建对比报告
    # 组装元信息
    meta_info = {
        '随机种子': 42,
        'ECE分箱': metrics_cfg.get('ECE', {}).get('num_bins', 15),
        '分箱策略': metrics_cfg.get('ECE', {}).get('bin_strategy', 'equal_width'),
        '校准来源': 'validation_set' if cal_logits is not None else 'fallback_from_test_split'
    }
    comparison_df = create_calibration_comparison(calibration_results, output_dir, meta_info)
    
    # 创建可视化
    create_calibration_visualizations(calibration_results, calibrators, output_dir)
    
    # 保存校准文件
    save_calibration_artifacts(calibrators, calibration_results, output_dir)
    
    # 输出关键结果
    logger.info("\\n=== 校准方法拟合完成 ===")
    
    if not comparison_df.empty:
        best_method = comparison_df.iloc[0]
        logger.info(f"最佳校准方法: {best_method['Method']}")
        logger.info(f"最佳ECE: {best_method['ECE']:.4f}")
        
        # 检查目标达成
        target_ece = 0.08
        if best_method['ECE'] <= target_ece:
            logger.info(f"✓ ECE目标达成 (≤ {target_ece})")
        else:
            logger.warning(f"✗ ECE目标未达成 (> {target_ece})")
        
        # 改进统计
        if 'ECE_Improvement' in best_method and best_method['ECE_Improvement'] > 0:
            logger.info(f"ECE最大改进: {best_method['ECE_Improvement']:.1f}%")
        
        logger.info("\\n校准方法排名 (按ECE):")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            logger.info(f"  {i}. {row['Method']}: ECE={row['ECE']:.4f}")
    
    logger.info("校准方法拟合完毕!")


if __name__ == "__main__":
    main()