#!/usr/bin/env python3
"""
实验一：路由性能测试脚本

功能:
1. 集成ComplexityAwareRouter
2. 测试路由准确率和覆盖率
3. 分析不同置信度阈值的影响
4. 生成路由性能报告

运行方式:
    python scripts/06_routing_test.py
"""

import os
import sys
import asyncio
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

# 尝试导入路由器（主项目实际路径：nano_graphrag.complexity.router）
try:
    from nano_graphrag.complexity.router import ComplexityAwareRouter
    ROUTER_AVAILABLE = True
except Exception:
    logger = logging.getLogger(__name__)
    logger.error("无法从 nano_graphrag.complexity.router 导入 ComplexityAwareRouter。请确保核心库已安装且导入路径正确。")
    ROUTER_AVAILABLE = False

from src.utils.metrics import RoutingMetrics
from src.utils.visualization import PerformancePlotter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/routing_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 移除模拟路由器：强制要求真实路由器


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


def load_best_classifier(output_dir: Path):
    """加载最佳分类器"""
    logger.info("加载最佳分类器...")
    
    # 根据评估结果选择最佳模型
    try:
        # 首先尝试加载校准后的模型（自动选择ECE最优的校准器）
        calibrators_dir = output_dir / 'models' / 'calibrators'
        
        if calibrators_dir.exists():
            # 1) 从对比表选择最优校准方法：兼顾准确率(降序)与ECE(升序)，跳过uncalibrated
            selected_method = 'temperature_scaling'
            try:
                comparison_file = output_dir / 'results' / 'calibration_comparison.csv'
                if comparison_file.exists():
                    comp_df = pd.read_csv(comparison_file)
                    comp_df = comp_df[comp_df['Method'] != 'uncalibrated']
                    if len(comp_df) > 0:
                        if 'Accuracy' not in comp_df.columns:
                            comp_df['Accuracy'] = np.nan
                        comp_df['Accuracy'] = comp_df['Accuracy'].fillna(0.0)
                        comp_df['ECE'] = comp_df['ECE'].fillna(1.0)
                        comp_df = comp_df.sort_values(['Accuracy', 'ECE'], ascending=[False, True])
                        selected_method = comp_df['Method'].iloc[0]
                        logger.info(f"自动选择最优校准方法(优先高准确率，其次低ECE): {selected_method}")
            except Exception as e:
                logger.warning(f"读取校准对比表失败，回退使用 {selected_method}: {e}")

            # 2) 按选择的方法加载对应校准器
            candidate_file = calibrators_dir / f"{selected_method}_calibrator.pkl"
            fallback_temp_file = calibrators_dir / 'temperature_scaling_calibrator.pkl'

            calibrator_file_to_use = None
            if candidate_file.exists():
                calibrator_file_to_use = candidate_file
            elif fallback_temp_file.exists():
                calibrator_file_to_use = fallback_temp_file
                logger.warning("未找到所选校准器文件，回退到 temperature_scaling_calibrator.pkl")

            if calibrator_file_to_use is not None:
                with open(calibrator_file_to_use, 'rb') as f:
                    best_calibrator = pickle.load(f)

                # 创建校准分类器包装
                class CalibratedClassifierWrapper:
                    def __init__(self, calibrator, base_logits):
                        # 中文说明：保存校准器与基础logits，以便对外提供 predict/predict_proba 接口
                        self.calibrator = calibrator
                        self.base_logits = base_logits  # 预先计算的logits
                    
                    def predict_proba(self, queries):
                        # 中文说明：此处依赖与测试集顺序一致的预存logits，不做实时前向
                        n_queries = len(queries)
                        if len(self.base_logits) < n_queries:
                            raise ValueError("预存logits数量小于查询数量，无法进行真实校准预测。")
                        selected_logits = self.base_logits[:n_queries]

                        # 应用校准
                        calibrated_probs = self.calibrator.transform(selected_logits)
                        return calibrated_probs
                    
                    def predict(self, queries):
                        probs = self.predict_proba(queries)
                        return np.argmax(probs, axis=1)

                # 尝试加载ModernBERT的logits
                results_file = output_dir / 'results' / 'modernbert_final_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    base_logits = np.array(results['test_logits'])
                else:
                    raise FileNotFoundError("缺少 outputs/results/modernbert_final_results.json，无法进行真实校准预测。")
                
                classifier = CalibratedClassifierWrapper(best_calibrator, base_logits)
                logger.info(f"加载校准后的分类器（{calibrator_file_to_use.name}）")
                return classifier
        
        # 如果没有校准器，尝试加载基础ModernBERT模型
        model_file = output_dir / 'models' / 'modernbert_best_model.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                classifier = pickle.load(f)
            logger.info("加载ModernBERT基础模型")
            return classifier
        
    except Exception as e:
        logger.warning(f"加载分类器失败: {e}")
    
    # 禁止模拟分类器
    raise RuntimeError("无法加载任何真实分类器或预存logits，请先完成 03 训练与 04 校准，或确保 outputs 目录完备。")


def initialize_router(classifier):
    """初始化路由器"""
    logger.info("初始化路由器...")

    if not ROUTER_AVAILABLE:
        raise RuntimeError("未能初始化真实路由器。")

    # 1) 实例化路由器（使用默认配置）
    router = ComplexityAwareRouter()

    # 2) 尝试以与校准阶段一致的方式，装配一个满足接口的“校准分类器适配器”
    try:
        output_dir = Path('outputs')
        calibrators_dir = output_dir / 'models' / 'calibrators'

        # 2.1 自动选择最优校准方法：兼顾准确率与ECE
        selected_method = 'temperature_scaling'
        comparison_file = output_dir / 'results' / 'calibration_comparison.csv'
        try:
            if comparison_file.exists():
                comp_df = pd.read_csv(comparison_file)
                comp_df = comp_df[comp_df['Method'] != 'uncalibrated']
                if len(comp_df) > 0:
                    if 'Accuracy' not in comp_df.columns:
                        comp_df['Accuracy'] = np.nan
                    comp_df['Accuracy'] = comp_df['Accuracy'].fillna(0.0)
                    comp_df['ECE'] = comp_df['ECE'].fillna(1.0)
                    comp_df = comp_df.sort_values(['Accuracy', 'ECE'], ascending=[False, True])
                    selected_method = comp_df['Method'].iloc[0]
                    logger.info(f"路由阶段自动选择最优校准方法(优先高准确率，其次低ECE): {selected_method}")
        except Exception as e:
            logger.warning(f"读取校准对比表失败，回退使用 {selected_method}: {e}")

        # 2.2 加载校准器与测试集logits
        candidate_file = calibrators_dir / f"{selected_method}_calibrator.pkl"
        fallback_temp_file = calibrators_dir / 'temperature_scaling_calibrator.pkl'
        calibrator_file_to_use = candidate_file if candidate_file.exists() else (
            fallback_temp_file if fallback_temp_file.exists() else None
        )

        if calibrator_file_to_use is None:
            logger.warning("未找到可用的校准器文件，将不装配校准分类器适配器。")
            logger.info("使用真实的ComplexityAwareRouter（内置未校准/规则）")
            return router

        with open(calibrator_file_to_use, 'rb') as f:
            calibrator = pickle.load(f)

        results_file = output_dir / 'results' / 'modernbert_final_results.json'
        if not results_file.exists():
            logger.warning("缺少 modernbert_final_results.json，无法使用预存logits进行校准预测。")
            return router

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        base_logits = np.array(results['test_logits'])

        # 2.3 构造满足 router 期望接口的适配器
        class CalibratedClassifierAdapter:
            """适配器：提供 is_available / predict_with_confidence 接口，供路由器调用。
            中文说明：该适配器顺序消费预存的测试集logits，并用选定的校准器生成概率与置信度。
            """

            def __init__(self, loaded_calibrator, logits_matrix: np.ndarray):
                self.loaded_calibrator = loaded_calibrator
                self.logits_matrix = logits_matrix
                self.current_index = 0
                self.id2label = {0: 'zero_hop', 1: 'one_hop', 2: 'multi_hop'}

            def is_available(self) -> bool:
                return True

            def predict_with_confidence(self, query: str):
                if self.current_index >= len(self.logits_matrix):
                    raise IndexError("预存logits已全部消费，无法继续进行校准预测。")

                raw_logits = self.logits_matrix[self.current_index]
                self.current_index += 1

                # 形状标准化并应用校准
                probs = self.loaded_calibrator.transform(raw_logits.reshape(1, -1))[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                complexity = self.id2label[pred_idx]

                # 返回路由器需要的三元组
                prob_dict = {
                    'zero_hop': float(probs[0]),
                    'one_hop': float(probs[1]),
                    'multi_hop': float(probs[2]),
                }
                return complexity, confidence, prob_dict

        router.classifier = CalibratedClassifierAdapter(calibrator, base_logits)
        logger.info(f"使用校准分类器适配器: {calibrator_file_to_use.name}")
        return router

    except Exception as e:
        logger.warning(f"装配校准分类器适配器失败，回退使用路由器默认行为: {e}")
        return router


def reset_router_classifier(router):
    """为路由器重新装配一个新的校准分类器适配器，避免预存logits被一次性消费完。
    中文说明：每次阈值测试前调用，保证 adapter 的 current_index 归零。
    """
    try:
        output_dir = Path('outputs')
        calibrators_dir = output_dir / 'models' / 'calibrators'

        # 选择最优校准器
        selected_method = 'temperature_scaling'
        comparison_file = output_dir / 'results' / 'calibration_comparison.csv'
        if comparison_file.exists():
            comp_df = pd.read_csv(comparison_file)
            comp_df = comp_df[comp_df['Method'] != 'uncalibrated']
            if len(comp_df) > 0:
                selected_method = comp_df.sort_values('ECE', ascending=True)['Method'].iloc[0]

        candidate_file = calibrators_dir / f"{selected_method}_calibrator.pkl"
        fallback_temp_file = calibrators_dir / 'temperature_scaling_calibrator.pkl'
        calibrator_file_to_use = candidate_file if candidate_file.exists() else (
            fallback_temp_file if fallback_temp_file.exists() else None
        )

        if calibrator_file_to_use is None:
            return

        with open(calibrator_file_to_use, 'rb') as f:
            calibrator = pickle.load(f)

        results_file = output_dir / 'results' / 'modernbert_final_results.json'
        if not results_file.exists():
            return

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        base_logits = np.array(results['test_logits'])

        class CalibratedClassifierAdapter:
            def __init__(self, loaded_calibrator, logits_matrix: np.ndarray):
                self.loaded_calibrator = loaded_calibrator
                self.logits_matrix = logits_matrix
                self.current_index = 0
                self.id2label = {0: 'zero_hop', 1: 'one_hop', 2: 'multi_hop'}

            def is_available(self) -> bool:
                return True

            def predict_with_confidence(self, query: str):
                if self.current_index >= len(self.logits_matrix):
                    raise IndexError("预存logits已全部消费，无法继续进行校准预测。")

                raw_logits = self.logits_matrix[self.current_index]
                self.current_index += 1

                probs = self.loaded_calibrator.transform(raw_logits.reshape(1, -1))[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                complexity = self.id2label[pred_idx]

                prob_dict = {
                    'zero_hop': float(probs[0]),
                    'one_hop': float(probs[1]),
                    'multi_hop': float(probs[2]),
                }
                return complexity, confidence, prob_dict

        router.classifier = CalibratedClassifierAdapter(calibrator, base_logits)
    except Exception:
        return


def test_routing_performance(router, X_test, y_test, test_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    """测试路由性能"""
    logger.info("测试路由性能...")
    
    routing_metrics = RoutingMetrics()
    all_results = []
    
    label_to_id = {'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}

    for threshold in test_thresholds:
        logger.info(f"测试置信度阈值: {threshold}")
        # 每个阈值前重置一次适配器，避免预存logits被耗尽
        reset_router_classifier(router)
        
        # 批量路由（兼容无 batch_route 的路由器）
        if hasattr(router, 'batch_route'):
            routing_results = router.batch_route(X_test, confidence_threshold=threshold)
        else:
            routing_results = _batch_route_via_async(router, X_test, confidence_threshold=threshold)
        
        # 提取预测结果
        predicted_labels = [r['predicted_complexity'] for r in routing_results]
        # 将标签映射为整数id
        predictions = [label_to_id[p] if isinstance(p, str) else int(p) for p in predicted_labels]
        confidences = [float(r['confidence']) for r in routing_results]
        route_modes = [r['route_mode'] for r in routing_results]
        fallback_usage = [r['used_fallback'] for r in routing_results]
        
        # 转为 numpy 数组
        predictions_np = np.array(predictions, dtype=int)

        # 计算路由准确率
        route_accuracy = routing_metrics.compute_route_accuracy(y_test, predictions_np)

        # 计算最优覆盖率：优先使用完整概率分布
        if 'probabilities' in routing_results[0]:
            probabilities = np.array([r['probabilities'] for r in routing_results], dtype=float)
        else:
            # 兜底：用置信度近似构造分布（不建议）。
            probabilities = np.array([[1-c, c, c] if p == 1 else [c, 1-c, c] if p == 0 else [c, c, 1-c]
                                     for p, c in zip(predictions_np, confidences)], dtype=float)
        probabilities_np = probabilities
        coverage_results = routing_metrics.compute_optimal_coverage(
            y_test, predictions_np, probabilities_np, threshold
        )
        
        # 分析路由模式分布
        distribution_results = routing_metrics.analyze_route_distribution(
            predictions_np, probabilities_np
        )
        
        # 汇总结果
        threshold_results = {
            'threshold': threshold,
            'route_accuracy': route_accuracy,
            'optimal_coverage': coverage_results['optimal_coverage'],
            'high_confidence_accuracy': coverage_results['high_confidence_accuracy'],
            'fallback_usage_rate': np.mean(fallback_usage),
            'avg_confidence': np.mean(confidences),
            'mode_distribution': distribution_results,
            'n_samples': len(X_test)
        }
        
        all_results.append(threshold_results)
        
        logger.info(f"阈值 {threshold}: 路由准确率={route_accuracy:.4f}, 覆盖率={coverage_results['optimal_coverage']:.4f}")
    
    return all_results


def _batch_route_via_async(router, queries, confidence_threshold: float):
    """在路由器无 batch_route 接口时，通过异步API适配实现批量路由。
    中文说明：调用 router.predict_complexity_detailed，并复用其路由退避逻辑选择模式。
    """
    # 中文说明：为确保每个测试阈值真正生效，这里显式更新路由器的阈值
    try:
        router.confidence_threshold = confidence_threshold
    except Exception:
        pass
    async def _route_one(q: str):
        # 使用详细预测接口
        result = await router.predict_complexity_detailed(q)
        complexity = result["complexity"]
        confidence = float(result["confidence"]) if "confidence" in result else 0.0
        candidate_modes = result.get("candidate_modes", ["naive"])  # 保底
        probs_dict = result.get("probabilities", None)

        used_fallback = False
        selected_mode = candidate_modes[0] if candidate_modes else "naive"

        # 复用路由器中的阈值退避规则
        try:
            if confidence < router.confidence_threshold and router.enable_fallback:
                used_fallback = True
                fallback_result = await router._rule_based_complexity(q)
                # 使用回退的复杂度作为最终预测标签，以反映阈值退避对准确率的真实影响
                if isinstance(fallback_result, dict) and 'complexity' in fallback_result:
                    complexity = fallback_result['complexity']
                    # 回退概率不可用时置零，保持键一致
                    fb_probs = fallback_result.get('probabilities', {}) or {}
                    probs_dict = {
                        'zero_hop': float(fb_probs.get('zero_hop', 0.0)),
                        'one_hop': float(fb_probs.get('one_hop', 0.0)),
                        'multi_hop': float(fb_probs.get('multi_hop', 0.0)),
                    }
                fb_candidates = fallback_result.get("candidate_modes", []) if isinstance(fallback_result, dict) else []
                if fb_candidates:
                    selected_mode = fb_candidates[0]
        except Exception:
            pass

        return {
            'predicted_complexity': complexity,
            'confidence': confidence,
            'route_mode': selected_mode,
            'used_fallback': used_fallback,
            'probabilities': [
                (probs_dict.get('zero_hop', 0.0) if probs_dict else 0.0),
                (probs_dict.get('one_hop', 0.0) if probs_dict else 0.0),
                (probs_dict.get('multi_hop', 0.0) if probs_dict else 0.0),
            ],
        }

    async def _run_all():
        tasks = [_route_one(q) for q in queries]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run_all())


def analyze_routing_results(routing_results):
    """分析路由结果"""
    logger.info("分析路由结果...")
    
    # 转换为DataFrame便于分析
    results_df = pd.DataFrame(routing_results)
    
    # 找到最佳阈值
    # 综合考虑准确率和覆盖率
    results_df['composite_score'] = (
        results_df['route_accuracy'] * 0.7 + 
        results_df['optimal_coverage'] * 0.3
    )
    
    best_threshold_idx = results_df['composite_score'].idxmax()
    best_threshold = results_df.loc[best_threshold_idx, 'threshold']
    best_results = results_df.loc[best_threshold_idx]
    
    logger.info(f"最佳阈值: {best_threshold}")
    logger.info(f"最佳路由准确率: {best_results['route_accuracy']:.4f}")
    logger.info(f"最佳覆盖率: {best_results['optimal_coverage']:.4f}")
    
    # 计算目标达成情况
    target_route_accuracy = 0.90
    achievable_thresholds = results_df[results_df['route_accuracy'] >= target_route_accuracy]
    
    analysis_results = {
        'best_threshold': best_threshold,
        'best_route_accuracy': best_results['route_accuracy'],
        'best_coverage': best_results['optimal_coverage'],
        'target_achieved': len(achievable_thresholds) > 0,
        'achievable_thresholds': achievable_thresholds['threshold'].tolist() if len(achievable_thresholds) > 0 else [],
        'threshold_analysis': results_df.to_dict('records')
    }
    
    return analysis_results


def create_routing_performance_report(routing_results, analysis_results, output_dir):
    """创建路由性能报告"""
    logger.info("创建路由性能报告...")
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    routing_file = results_dir / 'routing_performance_results.json'
    def _to_native_recursive(obj):
        # 深度转换 numpy / pandas 类型为原生 Python 类型
        if isinstance(obj, dict):
            return {k: _to_native_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native_recursive(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    serializable = {
        'routing_results': _to_native_recursive(routing_results),
        'analysis_results': _to_native_recursive(analysis_results)
    }

    with open(routing_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"保存路由结果: {routing_file}")
    
    # 创建结果表格
    results_df = pd.DataFrame(routing_results)
    results_df = results_df.round(4)
    
    table_file = results_dir / 'routing_threshold_analysis.csv'
    results_df.to_csv(table_file, index=False)
    logger.info(f"保存阈值分析表: {table_file}")
    
    # 生成Markdown报告
    report_file = results_dir / 'routing_performance_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 路由性能测试报告\\n\\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## 测试目标\\n\\n")
        f.write("- **★★ 路由准确率目标**: > 90% (系统实用性)\\n")
        f.write("- **覆盖率要求**: 平衡准确率和覆盖率\\n")
        f.write("- **置信度阈值优化**: 找到最佳阈值设置\\n\\n")
        
        f.write("## 阈值性能分析\\n\\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\\n\\n")
        
        f.write("## 关键发现\\n\\n")
        
        # 目标达成情况
        target_achieved = analysis_results['target_achieved']
        f.write(f"### 目标达成情况\\n")
        
        if target_achieved:
            f.write(f"✓ **路由准确率目标达成**: 在阈值 {analysis_results['achievable_thresholds']} 下达到 > 90%\\n")
        else:
            f.write(f"✗ **路由准确率目标未达成**: 最高准确率 {analysis_results['best_route_accuracy']:.4f}\\n")
        
        f.write("\\n")
        
        # 最佳配置
        f.write(f"### 推荐配置\\n")
        f.write(f"- **最佳置信度阈值**: {analysis_results['best_threshold']}\\n")
        f.write(f"- **路由准确率**: {analysis_results['best_route_accuracy']:.4f}\\n")
        f.write(f"- **覆盖率**: {analysis_results['best_coverage']:.4f}\\n\\n")
        
        # 阈值影响分析
        f.write("### 阈值影响分析\\n")
        f.write("| 阈值范围 | 特点 | 建议使用场景 |\\n")
        f.write("|----------|------|--------------|\\n")
        f.write("| 0.5-0.6 | 高覆盖率，中等准确率 | 需要处理所有查询的场景 |\\n")
        f.write("| 0.7-0.8 | 平衡准确率和覆盖率 | 大多数实际应用场景 |\\n")
        f.write("| 0.9+ | 高准确率，低覆盖率 | 高精度要求的关键应用 |\\n\\n")
        
        # 路由模式分析
        f.write("### 路由模式分布\\n")
        
        # 使用最佳阈值的结果
        best_result = next(r for r in routing_results if r['threshold'] == analysis_results['best_threshold'])
        mode_dist = best_result['mode_distribution']
        
        f.write(f"在最佳阈值 {analysis_results['best_threshold']} 下:\\n")
        for mode, stats in mode_dist.items():
            f.write(f"- **{mode}**: {stats['count']} 次 ({stats['percentage']:.1f}%), ")
            f.write(f"平均置信度 {stats['avg_confidence']:.3f}\\n")
        
        f.write("\\n")
        
        f.write("## 技术分析\\n\\n")
        
        # 置信度与准确率关系
        f.write("### 置信度-准确率关系\\n")
        f.write("观察到的模式:\\n")
        
        high_threshold_results = [r for r in routing_results if r['threshold'] >= 0.8]
        low_threshold_results = [r for r in routing_results if r['threshold'] <= 0.6]
        
        if high_threshold_results and low_threshold_results:
            high_acc = np.mean([r['route_accuracy'] for r in high_threshold_results])
            low_acc = np.mean([r['route_accuracy'] for r in low_threshold_results])
            
            f.write(f"- 高置信度阈值(≥0.8): 平均准确率 {high_acc:.4f}\\n")
            f.write(f"- 低置信度阈值(≤0.6): 平均准确率 {low_acc:.4f}\\n")
            f.write(f"- 准确率提升: {(high_acc - low_acc) * 100:.1f} 个百分点\\n\\n")
        
        # 回退机制效果
        f.write("### 回退机制效果\\n")
        avg_fallback = np.mean([r['fallback_usage_rate'] for r in routing_results])
        f.write(f"- 平均回退使用率: {avg_fallback:.2%}\\n")
        f.write(f"- 回退机制有效性: {'良好' if avg_fallback < 0.3 else '需要改进'}\\n\\n")
        
        f.write("## 结论与建议\\n\\n")
        
        if target_achieved:
            f.write("✓ **路由系统可用性验证成功**\\n")
            f.write("✓ 复杂度分类器能够有效支持查询路由\\n")
            f.write("✓ 置信度机制提供了良好的质量控制\\n\\n")
        else:
            f.write("⚠️ **路由准确率需要进一步优化**\\n")
            f.write("- 建议改进分类器训练策略\\n")
            f.write("- 考虑增强校准方法\\n")
            f.write("- 优化路由决策逻辑\\n\\n")
        
        f.write("### 部署建议\\n")
        f.write(f"1. **生产环境配置**: 使用置信度阈值 {analysis_results['best_threshold']}\\n")
        f.write("2. **监控指标**: 重点监控路由准确率和回退使用率\\n")
        f.write("3. **动态调整**: 根据实际查询分布调整阈值\\n")
        f.write("4. **故障保护**: 确保回退机制的可靠性\\n")
    
    logger.info(f"保存路由报告: {report_file}")


def create_routing_visualizations(routing_results, output_dir):
    """创建路由性能可视化"""
    logger.info("创建路由性能可视化...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 转换为DataFrame
        results_df = pd.DataFrame(routing_results)
        
        plotter = PerformancePlotter()
        
        # 路由性能曲线
        fig = plotter.plot_routing_performance_curves(
            results_df,
            title='Routing Performance vs Confidence Threshold'
        )
        
        curves_file = figures_dir / 'routing_performance_curves.png'
        fig.savefig(curves_file, dpi=300, bbox_inches='tight')
        logger.info(f"保存路由性能曲线: {curves_file}")
        
    except Exception as e:
        logger.warning(f"创建路由可视化失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="路由性能测试")
    parser.add_argument('--config', type=str, default='config/evaluation_config.yaml',
                       help='评估配置文件路径')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.5, 0.6, 0.7, 0.8, 0.9],
                       help='要测试的置信度阈值')
    args = parser.parse_args()
    
    logger.info("开始路由性能测试...")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载测试数据
    data_dir = Path('data/processed')
    X_test, y_test = load_test_data(data_dir)
    
    # 加载最佳分类器
    output_dir = Path('outputs')
    classifier = load_best_classifier(output_dir)
    
    # 初始化路由器
    router = initialize_router(classifier)
    
    # 测试路由性能
    routing_results = test_routing_performance(
        router, X_test, y_test, test_thresholds=args.thresholds
    )
    
    # 分析结果
    analysis_results = analyze_routing_results(routing_results)
    
    # 创建报告
    create_routing_performance_report(routing_results, analysis_results, output_dir)
    
    # 创建可视化
    create_routing_visualizations(routing_results, output_dir)
    
    # 输出关键结果
    logger.info("\\n=== 路由性能测试完成 ===")
    
    target_achieved = analysis_results['target_achieved']
    best_accuracy = analysis_results['best_route_accuracy']
    best_threshold = analysis_results['best_threshold']
    
    if target_achieved:
        logger.info(f"✓ 路由准确率目标达成: {best_accuracy:.4f} > 0.90")
        logger.info(f"✓ 推荐使用置信度阈值: {best_threshold}")
    else:
        logger.warning(f"✗ 路由准确率目标未达成: {best_accuracy:.4f} < 0.90")
        logger.info(f"当前最佳阈值: {best_threshold}")
    
    logger.info(f"最佳覆盖率: {analysis_results['best_coverage']:.4f}")
    logger.info("路由性能测试完毕!")


if __name__ == "__main__":
    main()