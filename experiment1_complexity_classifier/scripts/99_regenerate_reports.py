#!/usr/bin/env python3
"""
从现有产物重建报告与可视化，并将 ModernBERT 指标写入 baseline_models_summary_unified.csv。

使用：
  python scripts/99_regenerate_reports.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import pandas as pd


project_root = Path(__file__).parent.parent.parent.parent
experiment_root = Path(__file__).parent.parent

# 日志
log_dir = experiment_root / 'outputs' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'regenerate_reports.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def append_modernbert_to_summary(results_dir: Path) -> None:
    """将 ModernBERT 最终结果写入 baseline_models_summary_unified.csv。"""
    summary_csv = results_dir / 'baseline_models_summary_unified.csv'
    modern_path = results_dir / 'modernbert_final_results.json'
    if not modern_path.exists():
        logger.warning("未找到 modernbert_final_results.json，跳过汇总追加")
        return

    modern = load_json(modern_path)
    clf = modern.get('classification', {})
    cal = modern.get('calibration', {})

    new_row = {
        'Model': 'ModernBertClassifier',
        'Accuracy': float(clf.get('accuracy')) if clf.get('accuracy') is not None else None,
        'Macro-F1': float(clf.get('macro_f1')) if clf.get('macro_f1') is not None else None,
        'ECE': float(cal.get('ECE')) if cal.get('ECE') is not None else None,
        'Composite Score': (
            float(clf.get('accuracy')) - float(cal.get('ECE'))
            if clf.get('accuracy') is not None and cal.get('ECE') is not None else None
        ),
    }

    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        # 去重：若已存在 ModernBertClassifier，则更新为最新
        df = df[df['Model'] != 'ModernBertClassifier']
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # 排序：按 Composite Score 降序
    if 'Composite Score' in df.columns:
        df = df.sort_values('Composite Score', ascending=False, na_position='last')

    df.to_csv(summary_csv, index=False)
    logger.info("已写入/更新 ModernBERT 指标到 %s", summary_csv)


def regenerate_hparam_report(results_dir: Path, figures_dir: Path) -> None:
    """从超参 JSON 重建报告与可视化。"""
    # 动态加载 scripts/03_train_modernbert.py 中的函数（文件名以数字开头，不能直接 import）
    import importlib.util
    script_path = experiment_root / 'scripts' / '03_train_modernbert.py'
    spec = importlib.util.spec_from_file_location("train_modernbert", str(script_path))
    if spec is None or spec.loader is None:
        logger.warning("无法加载 03_train_modernbert.py，跳过报告重建")
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    create_hyperparameter_report = getattr(mod, 'create_hyperparameter_report', None)
    create_training_visualizations = getattr(mod, 'create_training_visualizations', None)
    if create_hyperparameter_report is None or create_training_visualizations is None:
        logger.warning("目标函数缺失，跳过报告重建")
        return

    hparam_json = results_dir / 'modernbert_hyperparameter_search.json'
    final_json = results_dir / 'modernbert_final_results.json'

    if not hparam_json.exists():
        logger.warning("未找到 modernbert_hyperparameter_search.json，跳过报告重建")
        return

    with open(hparam_json, 'r', encoding='utf-8') as f:
        hparam_results = json.load(f)

    # 兼容：如果保存的是列表里嵌套 cv_results 列表
    # 这里直接透传给 create_hyperparameter_report（已具备健壮性）
    create_hyperparameter_report(hparam_results, results_dir)

    # 组装 minimal 的 final_results 以便绘图函数运行
    if final_json.exists():
        final_results = load_json(final_json)
    else:
        final_results = {}

    try:
        create_training_visualizations(final_results, hparam_results, figures_dir)
        logger.info("已重建超参可视化")
    except Exception as e:
        logger.warning("可视化生成失败: %s", e)


def main() -> None:
    results_dir = experiment_root / 'outputs' / 'results'
    figures_dir = experiment_root / 'outputs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    append_modernbert_to_summary(results_dir)
    regenerate_hparam_report(results_dir, figures_dir)

    logger.info("重建完成: %s", datetime.now().isoformat())


if __name__ == '__main__':
    main()

