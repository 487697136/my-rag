#!/usr/bin/env python3
"""
实验一：ModernBERT训练脚本

功能:
1. 训练ModernBERT复杂度分类器
2. 超参数搜索和交叉验证
3. 记录训练过程和logits
4. 保存模型检查点

运行方式:
    python scripts/03_train_modernbert.py
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
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root))

from src.models.modernbert_classifier import ModernBertClassifier
from src.utils.metrics import ClassificationMetrics, CalibrationMetrics
from src.utils.visualization import PerformancePlotter

# 配置日志到实验输出目录（确保相对脚本的 outputs/logs 路径存在）
log_dir = experiment_root / 'outputs' / 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / 'modernbert_training.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 启用 TF32（性能优化，兼容不同 PyTorch 版本）
try:
    # 优先使用新 API 提示的设置
    torch.set_float32_matmul_precision('high')  # 使用 TensorFloat32 张量核
except Exception:
    # 回退到旧 API（未来版本将弃用，但对老环境友好）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        # 按警告建议设置卷积的 FP32 精度为 TF32
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    except Exception:
        pass

def _log_tf32_status() -> None:
    """打印 TF32 状态，便于在训练日志中明确记录。"""
    try:
        matmul_precision = getattr(torch, 'get_float32_matmul_precision', None)
        matmul_precision = matmul_precision() if matmul_precision else 'unknown'
    except Exception:
        matmul_precision = 'unknown'

    try:
        allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    except Exception:
        allow_tf32_matmul = 'unknown'

    try:
        allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    except Exception:
        allow_tf32_cudnn = 'unknown'

    try:
        conv_fp32_precision = torch.backends.cudnn.conv.fp32_precision
    except Exception:
        conv_fp32_precision = 'unknown'

    logger.info(
        "TF32 状态 | cuda_available=%s, matmul_precision=%s, "
        "cuda.matmul.allow_tf32=%s, cudnn.allow_tf32=%s, "
        "cudnn.conv.fp32_precision=%s",
        torch.cuda.is_available(), matmul_precision,
        allow_tf32_matmul, allow_tf32_cudnn, conv_fp32_precision,
    )

_log_tf32_status()


def load_processed_data(data_dir: Path) -> tuple:
    """加载处理后的数据 - 支持JSON格式"""
    logger.info("加载处理后的数据...")
    
    # 加载JSON格式数据
    with open(data_dir / 'train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(data_dir / 'calibration_data.json', 'r', encoding='utf-8') as f:
        calibration_data = json.load(f)
    with open(data_dir / 'test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 准备训练数据
    X_train = [item['query'] for item in train_data]
    y_train = [{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in train_data]
    
    # 准备校准数据
    X_cal = [item['query'] for item in calibration_data]
    y_cal = [{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in calibration_data]
    
    # 准备测试数据
    X_test = [item['query'] for item in test_data]
    y_test = [{'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}[item['complexity']] for item in test_data]
    
    logger.info(f"训练集: {len(X_train)} 条")
    logger.info(f"校准集: {len(X_cal)} 条")
    logger.info(f"测试集: {len(X_test)} 条")
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def create_hyperparameter_grid(config):
    """创建超参数搜索网格（与基线一致，保留 warmup 为固定默认值）"""
    search_config = config['training']['hyperparameter_search']

    param_grid = {
        'learning_rate': search_config['learning_rates'],
        'batch_size': search_config.get('batch_sizes', [16]),
        'weight_decay': search_config.get('weight_decay', [0.01]),
        'max_epochs': [3, 5],
    }

    grid = list(ParameterGrid(param_grid))
    logger.info(f"创建超参数网格，共 {len(grid)} 组合")

    return grid


def train_with_hyperparameters(X_train, y_train, X_val, y_val, params, config):
    """使用指定超参数训练模型"""
    logger.info(f"训练参数: {params}")
    
    try:
        # 初始化ModernBERT分类器
        modernbert_config = config['models']['modernbert']
        
        # 固定 warmup_steps 默认值（不纳入搜索）
        default_warmup = (
            config.get('training', {})
            .get('hyperparameter_search', {})
            .get('warmup_steps', [100])[0]
        )

        classifier = ModernBertClassifier(
            model_name=modernbert_config.get('model_name', 'answerdotai/ModernBERT-large'),
            num_classes=modernbert_config['num_classes'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            weight_decay=params['weight_decay'],
            warmup_steps=default_warmup,
            num_epochs=params.get('max_epochs', config['training']['max_epochs']),
            dropout=modernbert_config['dropout'],
            classifier_hidden_layers=modernbert_config.get('classifier_head', {}).get('hidden_layers'),
            early_stopping_patience=2,
            use_cross_validation=False,
            n_folds=1,
            random_state=config.get('random_seeds', {}).get('model_seed', 42),
            use_local_model=True,
            fallback_to_online=False
        )
        
        logger.info(
            "优化器计划: AdamW | lr=%.2e, weight_decay=%s, batch_size=%s, warmup_steps=%s, epochs=%s",
            params['learning_rate'], params['weight_decay'], params['batch_size'], default_warmup,
            params.get('max_epochs', config['training']['max_epochs'])
        )
        
        # 训练模型
        logger.info("开始训练ModernBERT...")
        classifier.fit(X_train, y_train, X_val, y_val)
        
        # 在验证集上评估
        val_predictions = classifier.predict(X_val)
        val_probabilities = classifier.predict_proba(X_val)
        val_logits = classifier.get_logits(X_val)
            
        # 计算验证分数
        val_accuracy = np.mean(val_predictions == y_val)
        
        # 计算ECE作为主要指标
        cal_metrics = CalibrationMetrics()
        val_calibration_results = cal_metrics.compute_all_calibration_metrics(
            np.array(y_val), val_probabilities
        )
        val_ece = val_calibration_results['ECE']
        
        # 综合分数（准确率和校准质量的平衡）
        composite_score = val_accuracy - val_ece  # 准确率高，ECE低为佳
        
        training_results = {
            'params': params,
            'val_accuracy': val_accuracy,
            'val_ece': val_ece,
            'composite_score': composite_score,
            'val_predictions': val_predictions,
            'val_probabilities': val_probabilities,
            'val_logits': val_logits,
            'training_history': getattr(classifier, 'training_history', []),
            'cv_scores': getattr(classifier, 'cv_scores', [])
        }
        
        logger.info(f"验证结果: 准确率={val_accuracy:.4f}, ECE={val_ece:.4f}, 综合分数={composite_score:.4f}")
        
        return classifier, training_results
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        # 默认严格模式：不再回落到模拟结果，确保实验使用真实数据与真实结果
        raise

    


def hyperparameter_search(X_train, y_train, config):
    """执行超参数搜索（使用 K 折交叉验证选择最佳参数）"""
    logger.info("开始超参数搜索...")

    param_grid = create_hyperparameter_grid(config)
    n_folds = config['training']['cross_validation']['n_folds']
    random_seed = config.get('random_seeds', {}).get('global_seed', 42)

    best_score = -float('inf')
    best_params = None
    all_results = []

    for i, params in enumerate(param_grid):
        logger.info(f"\\n--- 超参数组合 {i + 1}/{len(param_grid)} ---")
        cv_results = cross_validate_model(X_train, y_train, params, n_folds, random_seed, config)
        all_results.append({'params': params, 'cv_results': cv_results})

        if cv_results['composite_score'] > best_score:
            best_score = cv_results['composite_score']
            best_params = params
            logger.info(f"发现更好的参数! 综合分数: {best_score:.4f}")

    logger.info("\\n超参数搜索完成!")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳分数: {best_score:.4f}")

    return best_params, all_results


def final_training_and_evaluation(best_model, best_params, X_train, y_train, X_test, y_test, config):
    """使用最佳参数进行最终训练和评估"""
    logger.info("使用最佳参数进行最终训练...")
    
    if best_model is None:
        logger.error("未能获得训练好的模型，终止流程。")
        raise RuntimeError("ModernBERT 训练失败，未产生有效模型")
    
    try:
        # 在测试集上评估
        test_predictions = best_model.predict(X_test)
        test_probabilities = best_model.predict_proba(X_test)
        test_logits = best_model.get_logits(X_test)
    except Exception as e:
        logger.error(f"最终评估失败: {e}")
        raise
    
    # 计算详细评估指标
    clf_metrics = ClassificationMetrics()
    classification_results = clf_metrics.compute_all_metrics(
        np.array(y_test), test_predictions, test_probabilities
    )
    
    cal_metrics = CalibrationMetrics()
    calibration_results = cal_metrics.compute_all_calibration_metrics(
        np.array(y_test), test_probabilities, return_reliability_data=True
    )
    
    final_results = {
        'best_params': best_params,
        'classification': classification_results,
        'calibration': calibration_results,
        'test_predictions': test_predictions,
        'test_probabilities': test_probabilities,
        'test_logits': test_logits
    }
    
    # 记录最终结果
    logger.info("\\n=== 最终评估结果 ===")
    logger.info(f"测试准确率: {classification_results['accuracy']:.4f}")
    logger.info(f"宏F1分数: {classification_results['macro_f1']:.4f}")
    logger.info(f"ECE: {calibration_results['ECE']:.4f}")
    logger.info(f"MCE: {calibration_results['MCE']:.4f}")
    
    return best_model, final_results


def cross_validate_model(X_train, y_train, params, n_folds, random_seed, config):
    """与基线一致的交叉验证：K 折 + 折内 15% 验证集，评分=均值(accuracy) - 均值(ECE)"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    cv_accuracies = []
    cv_eces = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"  折 {fold_idx + 1}/{n_folds}")

        X_fold_train = [X_train[i] for i in tr_idx]
        y_fold_train = [y_train[i] for i in tr_idx]
        X_fold_test = [X_train[i] for i in te_idx]
        y_fold_test = [y_train[i] for i in te_idx]

        X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
            X_fold_train, y_fold_train, test_size=0.15, stratify=y_fold_train,
            random_state=random_seed + fold_idx
        )

        model, _ = train_with_hyperparameters(
            X_tr_inner, y_tr_inner, X_val_inner, y_val_inner, params, config
        )

        y_pred = model.predict(X_fold_test)
        y_proba = model.predict_proba(X_fold_test)

        acc = accuracy_score(y_fold_test, y_pred)
        cal_metrics = CalibrationMetrics()
        cal_res = cal_metrics.compute_all_calibration_metrics(
            np.array(y_fold_test), y_proba, return_reliability_data=False
        )
        ece = cal_res['ECE']

        cv_accuracies.append(acc)
        cv_eces.append(ece)

        try:
            import gc
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    mean_acc = float(np.mean(cv_accuracies)) if cv_accuracies else 0.0
    mean_ece = float(np.mean(cv_eces)) if cv_eces else 1.0
    composite = mean_acc - mean_ece

    return {
        'cv_accuracy': mean_acc,
        'cv_ece': mean_ece,
        'composite_score': composite,
        'cv_scores': cv_accuracies,
        'cv_ece_scores': cv_eces,
    }


def final_training_and_evaluation_cv(best_params, X_train, y_train, X_test, y_test, config):
    """使用最佳参数在全训练集（含 15% 验证）重训，然后在测试集评估（与基线一致）"""
    logger.info("使用最佳参数进行最终训练...")

    random_seed = config.get('random_seeds', {}).get('global_seed', 42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=random_seed
    )

    model, training_results = train_with_hyperparameters(
        X_tr, y_tr, X_val, y_val, best_params, config
    )

    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)
    test_logits = model.get_logits(X_test)

    clf_metrics = ClassificationMetrics()
    classification_results = clf_metrics.compute_all_metrics(
        np.array(y_test), test_predictions, test_probabilities
    )

    cal_metrics = CalibrationMetrics()
    calibration_results = cal_metrics.compute_all_calibration_metrics(
        np.array(y_test), test_probabilities, return_reliability_data=True
    )

    final_results = {
        'best_params': best_params,
        'classification': classification_results,
        'calibration': calibration_results,
        'test_predictions': test_predictions,
        'test_probabilities': test_probabilities,
        'test_logits': test_logits,
        # 新增：持久化验证集/校准集的 logits 与概率，供 04_calibration 优先使用
        'val_predictions': training_results.get('val_predictions'),
        'val_probabilities': training_results.get('val_probabilities'),
        'val_logits': training_results.get('val_logits'),
    }

    logger.info("\n=== 最终评估结果 ===")
    logger.info(f"测试准确率: {classification_results['accuracy']:.4f}")
    logger.info(f"宏F1分数: {classification_results['macro_f1']:.4f}")
    logger.info(f"ECE: {calibration_results['ECE']:.4f}")
    logger.info(f"MCE: {calibration_results['MCE']:.4f}")

    return model, final_results


def save_training_artifacts(model, final_results, hyperparameter_results, output_dir, config):
    """保存训练产物"""
    logger.info("保存训练产物...")
    
    # 创建输出目录
    model_dir = output_dir / 'models'
    results_dir = output_dir / 'results'
    figures_dir = output_dir / 'figures'
    
    for directory in [model_dir, results_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 保存最佳模型
    if model is not None:
        try:
            model_file = model_dir / 'modernbert_best_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"保存最佳模型: {model_file}")
        except Exception as e:
            logger.warning(f"保存模型失败: {e}")
    
    # 保存最终结果
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
    
    # 保存最终结果
    final_results_file = results_dir / 'modernbert_final_results.json'
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(final_results), f, indent=2, ensure_ascii=False)
    logger.info(f"保存最终结果: {final_results_file}")
    
    # 保存超参数搜索结果
    hyperparameter_file = results_dir / 'modernbert_hyperparameter_search.json'
    with open(hyperparameter_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(hyperparameter_results), f, indent=2, ensure_ascii=False)
    logger.info(f"保存超参数搜索结果: {hyperparameter_file}")
    
    # 创建超参数搜索报告
    create_hyperparameter_report(hyperparameter_results, results_dir)
    
    # 生成可视化
    try:
        create_training_visualizations(final_results, hyperparameter_results, figures_dir)
    except Exception as e:
        logger.warning(f"生成可视化失败: {e}")


def create_hyperparameter_report(hyperparameter_results, output_dir):
    """创建超参数搜索报告（兼容无 warmup_steps 与 CV 结构）

    期望输入：
    - 列表，每个元素包含：
      - params: { learning_rate, batch_size, weight_decay, [warmup_steps] }
      - cv_results: { cv_accuracy, cv_ece, composite_score }
    """
    import pandas as pd  # 局部导入以避免脚本入口阶段的环境依赖问题

    # 转换为DataFrame（健壮处理缺失字段）
    report_rows = []
    for result in hyperparameter_results:
        params = (result or {}).get('params', {}) or {}
        cv = (result or {}).get('cv_results', {}) or {}
        row = {
            'learning_rate': params.get('learning_rate'),
            'batch_size': params.get('batch_size'),
            'weight_decay': params.get('weight_decay'),
            'warmup_steps': params.get('warmup_steps'),
            'cv_accuracy': cv.get('cv_accuracy'),
            'cv_ece': cv.get('cv_ece'),
            'composite_score': cv.get('composite_score'),
        }
        report_rows.append(row)

    if not report_rows:
        logger.warning("超参数结果为空，跳过报告生成")
        return

    df = pd.DataFrame(report_rows)
    df = df.round(4)
    if 'composite_score' in df.columns:
        df = df.sort_values('composite_score', ascending=False, na_position='last')

    # 保存CSV
    csv_file = output_dir / 'hyperparameter_search_results.csv'
    df.to_csv(csv_file, index=False)

    # 生成Markdown报告
    report_file = output_dir / 'hyperparameter_search_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# ModernBERT超参数搜索报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 搜索结果汇总\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            # 最小降级：写出CSV风格文本
            f.write(df.to_csv(index=False))
        f.write("\n\n")

        # 最佳配置
        try:
            if not df.empty and pd.notna(df.iloc[0].get('composite_score')):
                f.write("## 最佳配置\n\n")
                best_row = df.iloc[0]
                f.write(f"- **学习率**: {best_row['learning_rate']}\n")
                f.write(f"- **批次大小**: {best_row['batch_size']}\n")
                f.write(f"- **权重衰减**: {best_row['weight_decay']}\n")
                f.write(f"- **预热步数**: {best_row.get('warmup_steps', '')}\n")
                if pd.notna(best_row.get('cv_accuracy')):
                    f.write(f"- **CV准确率**: {float(best_row['cv_accuracy']):.4f}\n")
                if pd.notna(best_row.get('cv_ece')):
                    f.write(f"- **CV ECE**: {float(best_row['cv_ece']):.4f}\n")
                if pd.notna(best_row.get('composite_score')):
                    f.write(f"- **综合分数**: {float(best_row['composite_score']):.4f}\n\n")
        except Exception:
            pass

        f.write("## 参数影响分析\n\n")

        # 学习率影响
        try:
            if 'learning_rate' in df.columns and 'composite_score' in df.columns:
                lr_groups = (
                    df.dropna(subset=['learning_rate', 'composite_score'])
                    .groupby('learning_rate')['composite_score']
                    .mean()
                    .sort_values(ascending=False)
                )
                f.write("### 学习率影响\n")
                for lr, score in lr_groups.items():
                    f.write(f"- {lr}: {score:.4f}\n")
                f.write("\n")
        except Exception:
            pass

        # 批次大小影响
        try:
            if 'batch_size' in df.columns and df['batch_size'].nunique() > 1:
                bs_groups = (
                    df.dropna(subset=['batch_size', 'composite_score'])
                    .groupby('batch_size')['composite_score']
                    .mean()
                    .sort_values(ascending=False)
                )
                f.write("### 批次大小影响\n")
                for bs, score in bs_groups.items():
                    f.write(f"- {bs}: {score:.4f}\n")
                f.write("\n")
        except Exception:
            pass

    logger.info(f"保存超参数报告: {report_file}")


def create_training_visualizations(final_results, hyperparameter_results, output_dir):
    """创建训练可视化"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 超参数搜索结果可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 提取数据（基于CV结果）
    learning_rates = [r['params']['learning_rate'] for r in hyperparameter_results]
    val_accuracies = [r['cv_results']['cv_accuracy'] for r in hyperparameter_results]
    val_eces = [r['cv_results']['cv_ece'] for r in hyperparameter_results]
    composite_scores = [r['cv_results']['composite_score'] for r in hyperparameter_results]
    
    # 学习率 vs 准确率
    axes[0, 0].scatter(learning_rates, val_accuracies, alpha=0.7)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].set_xscale('log')
    
    # 学习率 vs ECE
    axes[0, 1].scatter(learning_rates, val_eces, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Validation ECE')
    axes[0, 1].set_title('Learning Rate vs ECE')
    axes[0, 1].set_xscale('log')
    
    # 准确率 vs ECE
    axes[1, 0].scatter(val_accuracies, val_eces, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Validation Accuracy')
    axes[1, 0].set_ylabel('Validation ECE')
    axes[1, 0].set_title('Accuracy vs ECE Trade-off')
    
    # 综合分数分布
    axes[1, 1].hist(composite_scores, bins=10, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Composite Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Composite Score Distribution')
    
    plt.tight_layout()
    
    hyperparameter_fig_file = output_dir / 'modernbert_hyperparameter_analysis.png'
    plt.savefig(hyperparameter_fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"保存超参数分析图: {hyperparameter_fig_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ModernBERT训练")
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--skip-search', action='store_true',
                       help='跳过超参数搜索，使用默认参数')
    args = parser.parse_args()
    
    logger.info("开始ModernBERT训练...")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    data_dir = Path(config['data']['processed_data_dir'])
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data(data_dir)
    
    output_dir = Path('outputs')
    
    if args.skip_search:
        # 使用默认参数直接训练（内部切分验证集，不用校准集）
        logger.info("跳过超参数搜索，使用默认参数...")

        default_params = {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'weight_decay': 0.01,
            'max_epochs': 3,
        }

        final_model, final_results = final_training_and_evaluation_cv(
            default_params, X_train, y_train, X_test, y_test, config
        )
        hyperparameter_results = [
            {
                'params': default_params,
                'cv_results': {
                    'cv_accuracy': final_results['classification']['accuracy'],
                    'cv_ece': final_results['calibration']['ECE'],
                    'composite_score': final_results['classification']['accuracy'] - final_results['calibration']['ECE'],
                },
            }
        ]
        best_params = default_params
    else:
        # 执行超参数搜索（K 折）
        best_params, hyperparameter_results = hyperparameter_search(
            X_train, y_train, config
        )
    
    # 最终训练和评估
    final_model, final_results = final_training_and_evaluation_cv(
        best_params, X_train, y_train, X_test, y_test, config
    )
    
    # 保存训练产物
    save_training_artifacts(
        final_model, final_results, hyperparameter_results, output_dir, config
    )
    
    # 输出关键指标
    logger.info("\\n=== ModernBERT训练完成 ===")
    logger.info(f"最佳参数: {best_params}")
    
    if 'classification' in final_results:
        clf_results = final_results['classification']
        cal_results = final_results['calibration']
        
        logger.info(f"最终性能:")
        logger.info(f"  准确率: {clf_results['accuracy']:.4f}")
        logger.info(f"  宏F1: {clf_results['macro_f1']:.4f}")
        logger.info(f"  ECE: {cal_results['ECE']:.4f}")
        
        # 检查是否达到目标
        target_accuracy = 0.85
        target_ece = 0.08
        
        if clf_results['accuracy'] >= target_accuracy:
            logger.info(f"✓ 准确率达标 (≥ {target_accuracy})")
        else:
            logger.warning(f"✗ 准确率未达标 (< {target_accuracy})")
        
        if cal_results['ECE'] <= target_ece:
            logger.info(f"✓ ECE达标 (≤ {target_ece})")
        else:
            logger.warning(f"✗ ECE未达标 (> {target_ece})")
    
    logger.info("ModernBERT训练完毕!")


if __name__ == "__main__":
    main()