"""Snapshot utility for experiment1_complexity_classifier.

This script consolidates key result artifacts of Experiment 1 into a
timestamped snapshot directory under `outputs/snapshots/` and writes a
structured manifest for reproducibility and later paper support.

Usage:
    python scripts/99_make_snapshot.py

The script assumes it is invoked from the stage root directory:
`experiments/experiment1_complexity_classifier`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Any


# ------------------------------
# 数据类定义
# ------------------------------

@dataclass
class FileRecord:
    """单个文件的元信息记录。"""

    path: str
    exists: bool
    size_bytes: Optional[int]
    sha256: Optional[str]


@dataclass
class ThresholdPoint:
    """阈值-性能关键点记录。"""

    threshold: float
    route_accuracy: float
    optimal_coverage: float
    high_confidence_accuracy: float
    fallback_usage_rate: float


@dataclass
class CalibrationSummary:
    """校准方法摘要。"""

    best_method: Optional[str]
    table_rows: List[Dict[str, Any]]


@dataclass
class BaselineSummary:
    """基线模型摘要。"""

    entries: List[Dict[str, Any]]


@dataclass
class Manifest:
    """快照清单结构。"""

    stage: str
    timestamp: str
    dataset_samples: Optional[int]
    routing_best_threshold: Optional[float]
    routing_best_accuracy: Optional[float]
    routing_best_coverage: Optional[float]
    threshold_points: List[ThresholdPoint]
    calibration: CalibrationSummary
    baselines: BaselineSummary
    figures: List[str]
    copied_files: List[FileRecord]
    notes: str


# ------------------------------
# 工具函数
# ------------------------------

def compute_sha256(file_path: Path) -> Optional[str]:
    """计算文件的 SHA256 值。

    Args:
        file_path: 文件路径。

    Returns:
        十六进制 sha256 字符串或 None（若不存在）。
    """
    if not file_path.exists() or not file_path.is_file():
        return None
    h = sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> FileRecord:
    """安全复制文件并返回文件元信息。

    Args:
        src: 源文件路径。
        dst: 目标文件路径。

    Returns:
        FileRecord 记录对象。
    """
    exists = src.exists()
    if exists and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_bytes = src.stat().st_size
        digest = compute_sha256(dst)
    else:
        size_bytes = None
        digest = None
    return FileRecord(
        path=str(dst.as_posix()),
        exists=exists,
        size_bytes=size_bytes,
        sha256=digest,
    )


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """读取 JSON 文件为字典。"""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_threshold_points(results_json: Dict[str, Any]) -> List[ThresholdPoint]:
    """从路由结果 JSON 中抽取阈值关键点。"""
    points: List[ThresholdPoint] = []
    if not results_json:
        return points
    for item in results_json.get("analysis_results", {}).get(
        "threshold_analysis", []
    ):
        points.append(
            ThresholdPoint(
                threshold=float(item["threshold"]),
                route_accuracy=float(item["route_accuracy"]),
                optimal_coverage=float(item["optimal_coverage"]),
                high_confidence_accuracy=float(item["high_confidence_accuracy"]),
                fallback_usage_rate=float(item["fallback_usage_rate"]),
            )
        )
    return points


def parse_calibration_report(markdown_text: str) -> CalibrationSummary:
    """从校准 Markdown 文本粗略解析表格行和最优方法。

    说明: 为稳健起见，使用简单行匹配，而非完整 Markdown 解析。
    """
    rows: List[Dict[str, Any]] = []
    best_method: Optional[str] = None
    if not markdown_text:
        return CalibrationSummary(best_method=None, table_rows=rows)

    lines = [ln.strip() for ln in markdown_text.splitlines()]
    # 提取“最佳校准方法”行
    for idx, ln in enumerate(lines):
        if ln.startswith("- **最佳校准方法**") and ":" in ln:
            best_method = ln.split(":", 1)[1].strip()
            best_method = best_method.replace("**", "").strip()
            break

    # 提取简单 CSV 风格段落（Method,ECE,...）
    header_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if ln.startswith("Method,ECE,Brier") or ln.startswith("Method,ECE,MCE"):
            header_idx = i
            break
    if header_idx is not None:
        # 收集直到空行为止
        headers = [h.strip() for h in lines[header_idx].split(",")]
        for j in range(header_idx + 1, len(lines)):
            row_ln = lines[j]
            if not row_ln:
                break
            parts = [p.strip() for p in row_ln.split(",")]
            if len(parts) != len(headers):
                continue
            row = {headers[k]: parts[k] for k in range(len(headers))}
            rows.append(row)

    return CalibrationSummary(best_method=best_method, table_rows=rows)


def read_text(path: Path) -> str:
    """读取文本文件内容，若不存在返回空串。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ------------------------------
# 主流程
# ------------------------------

def main() -> int:
    """创建结果快照并生成 manifest。"""
    # 设定路径
    stage_root = Path.cwd()
    if not (stage_root / "outputs").exists():
        print("[ERROR] 请在实验阶段根目录运行此脚本。", file=sys.stderr)
        return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    snap_dir = stage_root / "outputs" / "snapshots" / f"exp1_{ts}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 关键文件路径
    results_dir = stage_root / "outputs" / "results"
    figures_dir = stage_root / "outputs" / "figures"

    files_to_copy = {
        "routing_threshold_analysis.csv": results_dir
        / "routing_threshold_analysis.csv",
        "routing_performance_results.json": results_dir
        / "routing_performance_results.json",
        "routing_performance_report.md": results_dir
        / "routing_performance_report.md",
        "calibration_report.md": results_dir / "calibration_report.md",
        "baseline_models_summary_unified.csv": results_dir
        / "baseline_models_summary_unified.csv",
        "routing_performance_curves.png": figures_dir
        / "routing_performance_curves.png",
    }

    copied: List[FileRecord] = []
    for name, src in files_to_copy.items():
        dst = snap_dir / name
        copied.append(safe_copy(src, dst))

    # 读取并解析路由结果
    routing_json_path = results_dir / "routing_performance_results.json"
    routing_json = read_json(routing_json_path) or {}

    analysis = routing_json.get("analysis_results", {})
    best_threshold = analysis.get("best_threshold")
    best_accuracy = analysis.get("best_route_accuracy")
    best_coverage = analysis.get("best_coverage")

    threshold_points = parse_threshold_points(routing_json)

    # 解析校准报告
    calib_md_path = results_dir / "calibration_report.md"
    calib_md = read_text(calib_md_path)
    calib_summary = parse_calibration_report(calib_md)

    # 读取基线摘要
    baseline_csv_path = results_dir / "baseline_models_summary_unified.csv"
    baseline_rows: List[Dict[str, Any]] = []
    if baseline_csv_path.exists():
        txt = read_text(baseline_csv_path)
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if lines:
            headers = [h.strip() for h in lines[0].split(",")]
            for ln in lines[1:]:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) != len(headers):
                    continue
                baseline_rows.append({headers[i]: parts[i] for i in range(len(headers))})

    # 样本数
    dataset_samples: Optional[int] = None
    if threshold_points:
        dataset_samples = read_json(routing_json_path).get("routing_results", [{}])[0].get(
            "n_samples"
        )

    # figures 相对路径
    figures: List[str] = []
    png_path = snap_dir / "routing_performance_curves.png"
    if png_path.exists():
        figures.append(str(Path("outputs/snapshots") / snap_dir.name / png_path.name))

    manifest = Manifest(
        stage="experiment1_complexity_classifier",
        timestamp=ts,
        dataset_samples=dataset_samples,
        routing_best_threshold=best_threshold,
        routing_best_accuracy=best_accuracy,
        routing_best_coverage=best_coverage,
        threshold_points=threshold_points,
        calibration=CalibrationSummary(
            best_method=calib_summary.best_method,
            table_rows=calib_summary.table_rows,
        ),
        baselines=BaselineSummary(entries=baseline_rows),
        figures=figures,
        copied_files=copied,
        notes=(
            "本快照聚合了阈值-覆盖-准确率权衡、校准对比与基线结果，用于后续实验对比与论文引用。"
        ),
    )

    # 写出 manifest.json
    manifest_path = snap_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)

    print(f"[OK] 结果快照已保存: {snap_dir}")
    print(f"      清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())