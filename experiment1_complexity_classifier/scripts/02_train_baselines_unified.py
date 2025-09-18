#!/usr/bin/env python3
"""
实验一：统一基线模型训练脚本
确保与ModernBERT使用相同的训练策略，保证实验公平性

功能:
1. 统一的超参数搜索策略
2. 相同的交叉验证方法
3. 一致的评估指标
4. 公平的对比基础

运行方式:
    python scripts/02_train_baselines_unified.py
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

# 设置环境变量以优化显存管理（新版变量名）
# 参考 PyTorch 警告：PYTORCH_CUDA_ALLOC_CONF 已弃用，请使用 PYTORCH_ALLOC_CONF
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 用于调试CUDA错误
from datetime import datetime
import torch
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score
import time
import hashlib

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root))

# 导入本地模型管理器
try:
    import importlib.util
    script_path = experiment_root / 'scripts' / '00_download_models.py'
    spec = importlib.util.spec_from_file_location("download_models", script_path)
    download_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_models)
    LocalModelManager = download_models.LocalModelManager
except Exception as e:
    # 如果导入失败，定义一个简单的占位符类
    print(f"警告：无法导入本地模型管理器: {e}")
    class LocalModelManager:
        def __init__(self, experiment_root):
            self.experiment_root = experiment_root
        def download_all_required_models(self, force_redownload=False):
            return {}
        def is_model_downloaded(self, model_name):
            return False
        def list_downloaded_models(self):
            return {}

from src.models.base_classifiers import (
    RandomClassifier,
    RuleBasedClassifier,
    BertClassifier,
    RobertaClassifier
)
from src.utils.metrics import ClassificationMetrics, CalibrationMetrics
from src.utils.visualization import PerformancePlotter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/baseline_training_unified.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedTrainingFramework:
    """统一训练框架 - 确保所有模型使用相同的训练策略，支持断点续训和本地模型管理"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.random_seed = config.get('random_seeds', {}).get('global_seed', 42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.model_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        
        for dir_path in [self.model_dir, self.results_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 检查点文件
        self.checkpoint_file = self.checkpoint_dir / 'training_progress.json'
        self.training_progress = self.load_training_progress()
        
        # 本地模型管理器
        self.local_model_manager = LocalModelManager(experiment_root)
        
        # 本地模型配置
        self.use_local_models = config.get('local_models', {}).get('enabled', True)
        self.fallback_to_online = config.get('local_models', {}).get('fallback_to_online', True)
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def load_training_progress(self):
        """加载训练进度"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"加载训练进度: {len(progress.get('completed_models', []))} 个模型已完成")
                return progress
            except Exception as e:
                logger.warning(f"加载训练进度失败: {e}，将重新开始")
        
        return {
            'completed_models': [],
            'failed_models': [],
            'training_start_time': None,
            'last_update_time': None
        }
    
    def save_training_progress(self):
        """保存训练进度"""
        self.training_progress['last_update_time'] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_progress, f, indent=2, ensure_ascii=False)
            logger.debug("训练进度已保存")
        except Exception as e:
            logger.error(f"保存训练进度失败: {e}")
    
    def is_model_completed(self, model_name: str) -> bool:
        """检查模型是否已完成训练"""
        return model_name in self.training_progress['completed_models']
    
    def mark_model_completed(self, model_name: str):
        """标记模型训练完成"""
        if model_name not in self.training_progress['completed_models']:
            self.training_progress['completed_models'].append(model_name)
        self.save_training_progress()
    
    def mark_model_failed(self, model_name: str, error_msg: str):
        """标记模型训练失败"""
        failure_info = {
            'model_name': model_name,
            'error': str(error_msg),
            'timestamp': datetime.now().isoformat()
        }
        self.training_progress['failed_models'].append(failure_info)
        self.save_training_progress()
    
    def load_existing_model_result(self, model_name: str):
        """加载已存在的模型结果"""
        model_file = self.model_dir / f"{model_name}_unified.pkl"
        results_file = self.results_dir / f"{model_name}_unified_results.json"
        
        if not (model_file.exists() and results_file.exists()):
            return None
        
        try:
            # 加载模型
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 加载结果
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"[加载] 已完成的模型: {model_name}")
            
            # 重构结果格式以匹配训练函数的返回值
            return {
                'model': model,
                'evaluation': results['evaluation'],
                'best_params': results.get('best_params', {}),
                'cv_results': results.get('cv_results', {}),
                'predictions': None,  # 这些会在需要时重新计算
                'probabilities': None,
                'logits': None
            }
            
        except Exception as e:
            logger.warning(f"加载已存在模型失败 {model_name}: {e}")
            return None
    
    def save_single_model_result(self, model_name: str, result: dict):
        """立即保存单个模型的结果"""
        try:
            # 保存模型
            if result['model'] is not None:
                model_file = self.model_dir / f"{model_name}_unified.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(result['model'], f)
                logger.info(f"[保存] 模型: {model_file}")
            
            # 保存结果
            results_file = self.results_dir / f"{model_name}_unified_results.json"
            
            serializable_result = {
                'model_name': model_name,
                'evaluation': result['evaluation'],
                'best_params': result.get('best_params', {}),
                'cv_results': result.get('cv_results', {}),
                'training_metadata': {
                    'framework': 'unified',
                    'timestamp': datetime.now().isoformat(),
                    'device': self.device,
                    'random_seed': self.random_seed
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[保存] 结果: {results_file}")
            
            # 标记完成
            self.mark_model_completed(model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"保存模型结果失败 {model_name}: {e}")
            self.mark_model_failed(model_name, str(e))
            return False
    
    def load_processed_data(self, data_dir: Path) -> tuple:
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
    
    def ensure_models_available(self, force_download: bool = False) -> bool:
        """确保训练所需的模型已下载到本地"""
        if not self.use_local_models:
            logger.info("🌐 本地模型功能已禁用，将使用在线模型")
            return True
        
        logger.info("🔍 检查本地模型可用性...")
        
        # 需要的模型列表
        required_models = ['bert-base-cased', 'roberta-large']
        missing_models = []
        
        # 检查每个模型是否已下载
        for model_name in required_models:
            if not self.local_model_manager.is_model_downloaded(model_name):
                missing_models.append(model_name)
                logger.warning(f"❌ 模型未下载: {model_name}")
            else:
                logger.info(f"✅ 模型已下载: {model_name}")
        
        # 如果有缺失的模型，尝试下载
        if missing_models or force_download:
            if missing_models:
                logger.info(f"📥 需要下载 {len(missing_models)} 个模型: {missing_models}")
            
            if force_download:
                logger.info("🔄 强制重新下载所有模型...")
            
            # 下载缺失的模型
            download_results = self.local_model_manager.download_all_required_models(
                force_redownload=force_download
            )
            
            # 检查下载结果
            all_downloaded = all(download_results.values())
            
            if all_downloaded:
                logger.info("✅ 所有模型已成功下载到本地")
                return True
            else:
                failed_models = [model for model, success in download_results.items() if not success]
                if self.fallback_to_online:
                    logger.warning(f"⚠️ 部分模型下载失败 {failed_models}，将在训练时回退到在线模型")
                    return True
                else:
                    logger.error(f"❌ 模型下载失败且禁用在线回退: {failed_models}")
                    return False
        else:
            logger.info("✅ 所有所需模型已在本地可用")
            return True
    
    def create_hyperparameter_grid(self, model_type: str):
        """创建超参数搜索网格 - 与ModernBERT一致"""
        if model_type in ['random', 'rule_based']:
            # 简单模型无需超参数搜索
            return [{}]
        
        # Transformer模型使用与ModernBERT相同的搜索空间
        search_config = self.config['training']['hyperparameter_search']
        
        # 基线模型使用相对保守的搜索空间（避免过度优化）
        param_grid = {
            'learning_rate': search_config['learning_rates'],
            'batch_size': search_config.get('batch_sizes', [16]),
            'max_epochs': [3, 5],  # 比ModernBERT少一些epoch
            'weight_decay': search_config.get('weight_decay', [0.01])
        }
        
        grid = list(ParameterGrid(param_grid))
        logger.info(f"{model_type} 超参数网格，共 {len(grid)} 组合")
        
        return grid
    
    def cross_validate_model(self, model_class, X_train, y_train, params, n_folds=5):
        """改进的交叉验证 - 支持训练过程监控和早停"""
        from sklearn.model_selection import train_test_split
        
        logger.info(f"进行{n_folds}折嵌套交叉验证...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        cv_scores = []
        cv_ece_scores = []
        fold_histories = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  折 {fold + 1}/{n_folds}")
            
            # 获取fold的训练和测试数据
            X_fold_train = [X_train[i] for i in train_idx]
            y_fold_train = [y_train[i] for i in train_idx]
            X_fold_test = [X_train[i] for i in test_idx]
            y_fold_test = [y_train[i] for i in test_idx]
            
            # 在fold训练集内部再分出验证集（用于训练监控）
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                X_fold_train, y_fold_train, 
                test_size=0.15,  # 15%作为验证集
                stratify=y_fold_train,
                random_state=self.random_seed + fold
            )
            
            logger.info(f"    训练集: {len(X_train_inner)}, 验证集: {len(X_val_inner)}, 测试集: {len(X_fold_test)}")
            
            try:
                # 训练模型
                model = model_class(**params.get('model_params', {}))
                
                if hasattr(model, 'fit_with_params'):
                    # 为支持超参数的模型添加验证数据
                    params_with_val = params.copy()
                    params_with_val['validation_data'] = (X_val_inner, y_val_inner)
                    params_with_val['early_stopping'] = True
                    params_with_val['patience'] = 2  # 早停耐心值
                    
                    # 训练模型并获取历史
                    history = model.fit_with_params(X_train_inner, y_train_inner, params_with_val)
                    
                    # 验证返回的是否为有效的history字典
                    if isinstance(history, dict) and all(key in history for key in ['train_loss', 'train_acc', 'val_acc']):
                        fold_histories.append(history)
                    else:
                        logger.warning(f"fit_with_params返回了无效的history对象: {type(history)}")
                        fold_histories.append(None)
                else:
                    # 简单模型
                    model.fit(X_train_inner, y_train_inner)
                    fold_histories.append(None)
                
                # 在fold测试集上评估（这是真正的交叉验证评估）
                y_pred = model.predict(X_fold_test)
                y_proba = model.predict_proba(X_fold_test)
                
                # 计算准确率
                accuracy = accuracy_score(y_fold_test, y_pred)
                cv_scores.append(accuracy)
                
                # 计算ECE
                cal_metrics = CalibrationMetrics()
                calibration_results = cal_metrics.compute_all_calibration_metrics(
                    y_fold_test, y_proba, return_reliability_data=False
                )
                ece = calibration_results['ECE']
                cv_ece_scores.append(ece)
                
                logger.info(f"    最终准确率: {accuracy:.4f}, ECE: {ece:.4f}")
                
            except Exception as e:
                logger.error(f"    折 {fold + 1} 训练失败: {e}")
                # 不要使用假数据！让失败真正失败
                raise RuntimeError(f"交叉验证折 {fold + 1} 训练失败: {e}") from e
            finally:
                # 清理显存与临时对象，避免后续折 OOM
                try:
                    import gc
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        
        mean_accuracy = np.mean(cv_scores)
        mean_ece = np.mean(cv_ece_scores)
        composite_score = mean_accuracy - mean_ece  # 与ModernBERT一致的评分标准
        
        logger.info(f"  交叉验证完成 - 平均准确率: {mean_accuracy:.4f}, 平均ECE: {mean_ece:.4f}")
        
        # 可视化训练历史（如果有的话）
        if any(h is not None for h in fold_histories):
            self._plot_training_curves(fold_histories)
        
        return {
            'cv_accuracy': mean_accuracy,
            'cv_ece': mean_ece,
            'composite_score': composite_score,
            'cv_scores': cv_scores,
            'cv_ece_scores': cv_ece_scores,
            'fold_histories': fold_histories  # 新增：每个fold的训练历史
        }
    
    def _plot_training_curves(self, fold_histories):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            # 创建输出目录
            os.makedirs('outputs/training_curves', exist_ok=True)
            
            # 过滤有效的history
            valid_histories = []
            for i, history in enumerate(fold_histories):
                if history is None:
                    continue
                if not isinstance(history, dict):
                    logger.warning(f"fold {i} history不是字典类型: {type(history)}")
                    continue
                if not all(key in history for key in ['train_loss', 'train_acc']):
                    logger.warning(f"fold {i} history缺少必要字段: {list(history.keys()) if isinstance(history, dict) else 'not a dict'}")
                    continue
                valid_histories.append((i, history))
            
            if not valid_histories:
                logger.warning("没有有效的训练历史可绘制")
                return
            
            # 为每个fold绘制曲线
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Training Curves Across Folds', fontsize=16)
            
            for plot_idx, (fold_idx, history) in enumerate(valid_histories):
                if plot_idx >= 6:  # 最多显示6个fold
                    break
                    
                row = plot_idx // 3
                col = plot_idx % 3
                ax = axes[row, col]
                
                epochs = range(1, len(history['train_loss']) + 1)
                
                # 绘制损失曲线
                ax2 = ax.twinx()
                line1 = ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
                line2 = ax.plot(epochs, history['train_acc'], 'g-', label='Train Acc')
                if history['val_acc']:
                    line3 = ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss / Accuracy')
                ax.set_title(f'Fold {fold_idx + 1}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏空的子图
            for i in range(len(fold_histories), 6):
                row = i // 3
                col = i % 3
                if row < 2:
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('outputs/training_curves/fold_training_curves.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("训练曲线已保存到 outputs/training_curves/fold_training_curves.png")
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过训练曲线绘制")
        except Exception as e:
            logger.warning(f"绘制训练曲线时出错: {e}")
    
    def train_random_classifier(self, X_train, y_train, X_cal, y_cal, X_test, y_test):
        """训练随机分类器"""
        model_name = 'random'
        logger.info("=== 训练随机分类器 ===")
        
        # 检查是否已完成训练
        if self.is_model_completed(model_name):
            logger.info("🔄 随机分类器已训练完成，加载已有结果...")
            return self.load_existing_model_result(model_name)
        
        try:
            # 无需超参数搜索
            clf = RandomClassifier(
                num_classes=3,
                random_state=self.random_seed
            )
            clf.fit(X_train, y_train)
            
            # 评估
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            logits = clf.get_logits(X_test)
            
            result = {
                'model': clf,
                'predictions': y_pred,
                'probabilities': y_proba,
                'logits': logits,
                'best_params': {},
                'cv_results': None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"随机分类器训练失败: {e}")
            self.mark_model_failed(model_name, str(e))
            raise
    
    def train_rule_based_classifier(self, X_train, y_train, X_cal, y_cal, X_test, y_test):
        """训练规则分类器"""
        model_name = 'rule_based'
        logger.info("=== 训练规则分类器 ===")
        
        # 检查是否已完成训练
        if self.is_model_completed(model_name):
            logger.info("🔄 规则分类器已训练完成，加载已有结果...")
            return self.load_existing_model_result(model_name)
        
        try:
            rule_config = self.config['models']['baselines']['rule_based']
            clf = RuleBasedClassifier(rule_config)
            clf.fit(X_train, y_train)
            
            # 评估
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            logits = clf.get_logits(X_test)
            
            result = {
                'model': clf,
                'predictions': y_pred,
                'probabilities': y_proba,
                'logits': logits,
                'best_params': rule_config,
                'cv_results': None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"规则分类器训练失败: {e}")
            self.mark_model_failed(model_name, str(e))
            raise
    
    def train_transformer_model(self, model_class, model_name, X_train, y_train, X_cal, y_cal, X_test, y_test):
        """训练Transformer模型 - 使用与ModernBERT相同的策略"""
        # 将模型名称映射到标准化的键
        model_key_map = {
            'BERT-Base': 'bert_base',
            'RoBERTa-Large': 'roberta_large',
            'ModernBERT': 'modernbert'
        }
        model_key = model_key_map.get(model_name, model_name.lower().replace('-', '_'))
        
        logger.info(f"=== 训练 {model_name} ===")
        
        # 检查是否已完成训练
        if self.is_model_completed(model_key):
            logger.info(f"🔄 {model_name} 已训练完成，加载已有结果...")
            return self.load_existing_model_result(model_key)
        
        try:
            # 超参数搜索
            param_grid = self.create_hyperparameter_grid(model_name.lower())
            
            best_score = -float('inf')
            best_params = None
            best_model = None
            best_cv_results = None
            all_results = []
            
            for i, params in enumerate(param_grid):
                logger.info(f"\n--- {model_name} 超参数组合 {i+1}/{len(param_grid)} ---")
                logger.info(f"参数: {params}")
                
                try:
                    # 准备模型参数
                    model_config = self.config['models']['baselines'][model_name.lower().replace('-', '_')]
                    model_params = {
                        'model_name': model_config['model_name'],
                        'num_classes': model_config['num_classes'],
                        'device': self.device,
                        'use_local_model': self.use_local_models,
                        'fallback_to_online': self.fallback_to_online
                    }
                    params['model_params'] = model_params
                    
                    # 交叉验证
                    cv_results = self.cross_validate_model(
                        model_class, X_train, y_train, params,
                        n_folds=self.config['training']['cross_validation']['n_folds']
                    )
                    
                    all_results.append({
                        'params': params,
                        'cv_results': cv_results
                    })
                    
                    # 更新最佳模型
                    if cv_results['composite_score'] > best_score:
                        best_score = cv_results['composite_score']
                        best_params = params
                        best_cv_results = cv_results
                        logger.info(f"发现更好的参数! 综合分数: {best_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"参数组合训练失败: {e}")
                    continue
        
            # 使用最佳参数重新训练
            if best_params is not None:
                logger.info(f"\n使用最佳参数重新训练 {model_name}...")
                logger.info(f"最佳参数: {best_params}")
                
                try:
                    model = model_class(**best_params['model_params'])
                    
                    if hasattr(model, 'fit_with_params'):
                        model.fit_with_params(X_train, y_train, best_params)
                    else:
                        model.fit(X_train, y_train)
                    
                    # 测试集评估
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)
                    logits = model.get_logits(X_test)
                    
                    best_model = model
                    
                except Exception as e:
                    logger.error(f"最终训练失败: {e}")
                    self.mark_model_failed(model_key, str(e))
                    raise
            else:
                error_msg = f"{model_name} 训练完全失败，所有参数组合都失败"
                logger.error(error_msg)
                self.mark_model_failed(model_key, error_msg)
                raise RuntimeError(error_msg)
            
            result = {
                'model': best_model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'logits': logits,
                'best_params': best_params,
                'cv_results': best_cv_results,
                'all_results': all_results
            }
            
            return result
            
        except Exception as e:
            logger.error(f"{model_name} 训练失败: {e}")
            self.mark_model_failed(model_key, str(e))
            raise
    
    def evaluate_model(self, y_true, y_pred, y_proba, model_name):
        """统一的模型评估 - 与ModernBERT完全一致"""
        logger.info(f"评估 {model_name} 性能...")
        
        # 分类指标
        clf_metrics = ClassificationMetrics()
        classification_results = clf_metrics.compute_all_metrics(y_true, y_pred, y_proba)
        
        # 校准指标
        cal_metrics = CalibrationMetrics()
        calibration_results = cal_metrics.compute_all_calibration_metrics(
            y_true, y_proba, return_reliability_data=True
        )
        
        # 合并结果
        results = {
            'model_name': model_name,
            'classification': classification_results,
            'calibration': calibration_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 记录关键指标
        logger.info(f"{model_name} 结果:")
        logger.info(f"  准确率: {classification_results['accuracy']:.4f}")
        logger.info(f"  宏F1: {classification_results['macro_f1']:.4f}")
        logger.info(f"  ECE: {calibration_results['ECE']:.4f}")
        
        return results


def main():
    """主函数 - 支持断点续训"""
    parser = argparse.ArgumentParser(description='统一基线模型训练 - 支持断点续训')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--resume', action='store_true', help='从断点继续训练')
    parser.add_argument('--force_retrain', type=str, nargs='+', help='强制重新训练指定模型')
    parser.add_argument('--download_models', action='store_true', help='强制重新下载所需模型')
    parser.add_argument('--use_online_models', action='store_true', help='禁用本地模型，直接使用在线模型')
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理命令行参数对配置的覆盖
    if args.use_online_models:
        logger.info("🌐 命令行指定使用在线模型，禁用本地模型功能")
        config.setdefault('local_models', {})['enabled'] = False
    
    # 初始化框架（现在需要output_dir参数）
    framework = UnifiedTrainingFramework(config, args.output_dir)
    
    # 记录训练开始时间
    if framework.training_progress['training_start_time'] is None:
        framework.training_progress['training_start_time'] = datetime.now().isoformat()
        framework.save_training_progress()
    
    # 确保所需模型已下载到本地
    logger.info(f"\n{'='*70}")
    logger.info("[准备] 检查和下载所需模型...")
    logger.info(f"{'='*70}")
    
    models_available = framework.ensure_models_available(force_download=args.download_models)
    if not models_available:
        logger.error("❌ 无法确保模型可用性，退出训练")
        sys.exit(1)
    
    # 加载数据
    data_dir = Path(config['data']['processed_data_dir'])
    X_train, y_train, X_cal, y_cal, X_test, y_test = framework.load_processed_data(data_dir)
    
    # 定义所有要训练的模型
    models_to_train = [
        ('random', 'RandomClassifier', framework.train_random_classifier),
        ('rule_based', 'RuleBasedClassifier', framework.train_rule_based_classifier),
        ('bert_base', 'BertClassifier', lambda *args: framework.train_transformer_model(BertClassifier, 'BERT-Base', *args)),
        ('roberta_large', 'RobertaClassifier', lambda *args: framework.train_transformer_model(RobertaClassifier, 'RoBERTa-Large', *args))
    ]
    
    # 处理强制重新训练
    if args.force_retrain:
        for model_name in args.force_retrain:
            if model_name in framework.training_progress['completed_models']:
                framework.training_progress['completed_models'].remove(model_name)
                logger.info(f"🔄 强制重新训练: {model_name}")
        framework.save_training_progress()
    
    # 训练所有基线模型
    model_results = {}
    total_models = len(models_to_train)
    completed_count = len(framework.training_progress['completed_models'])
    
    logger.info(f"\n[开始] 训练流程 - 总共 {total_models} 个模型，已完成 {completed_count} 个")
    
    for model_key, model_display_name, train_func in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"[处理] 模型: {model_display_name} ({model_key})")
        logger.info(f"[进度] {len(model_results)+1}/{total_models}")
        
        try:
            start_time = time.time()
            
            # 训练模型
            result = train_func(X_train, y_train, X_cal, y_cal, X_test, y_test)
            
            # 如果需要重新计算预测（从加载的模型）
            if result['predictions'] is None and result['model'] is not None:
                logger.info("🔄 重新计算预测结果...")
                result['predictions'] = result['model'].predict(X_test)
                result['probabilities'] = result['model'].predict_proba(X_test)
                result['logits'] = result['model'].get_logits(X_test)
            
            # 评估模型
            evaluation = framework.evaluate_model(
                y_test, result['predictions'], result['probabilities'], model_display_name
            )
            
            # 完整结果
            complete_result = {
                **result,
                'evaluation': evaluation
            }
            
            # 立即保存结果
            if framework.save_single_model_result(model_key, complete_result):
                model_results[model_key] = complete_result
                    
                training_time = time.time() - start_time
                logger.info(f"[完成] {model_display_name} 训练完成! 用时: {training_time:.1f}秒")
                logger.info(f"[结果] 准确率: {evaluation['classification']['accuracy']:.4f}")
                logger.info(f"[结果] ECE: {evaluation['calibration']['ECE']:.4f}")
            else:
                logger.error(f"❌ {model_display_name} 保存失败")
                
        except Exception as e:
            logger.error(f"❌ {model_display_name} 训练失败: {e}")
            # 继续训练下一个模型，不中断整个流程
            continue
    
    # 生成汇总报告
    logger.info(f"\n{'='*70}")
    logger.info("[完成] 基线模型训练流程完成!")
    logger.info(f"{'='*70}")
    
    if model_results:
        summary_data = []
        for model_name, result in model_results.items():
            eval_result = result['evaluation']
            summary_data.append({
                'Model': eval_result['model_name'],
                'Accuracy': f"{eval_result['classification']['accuracy']:.4f}",
                'Macro-F1': f"{eval_result['classification']['macro_f1']:.4f}",
                'ECE': f"{eval_result['calibration']['ECE']:.4f}",
                'Composite Score': f"{eval_result['classification']['accuracy'] - eval_result['calibration']['ECE']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n[汇总] 训练结果:")
        print(summary_df.to_string(index=False))
        
        # 保存汇总
        # 统一结果文件命名，供后续评估脚本使用
        summary_file = framework.results_dir / 'baseline_models_summary_unified.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"📁 汇总结果保存至: {summary_file}")

        # 额外输出单模型结果文件，命名与 05_evaluation.py 读取保持一致
        try:
            for model_key, complete_result in model_results.items():
                mapped = {
                    'random': 'random',
                    'rule_based': 'rule_based',
                    'bert_base': 'bert',
                    'roberta_large': 'roberta'
                }.get(model_key, model_key)
                single_file = framework.results_dir / f"{mapped}_results.json"
                with open(single_file, 'w', encoding='utf-8') as f:
                    json.dump(complete_result['evaluation'], f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"[保存] 评估摘要: {single_file}")
        except Exception as e:
            logger.warning(f"导出评估摘要文件失败: {e}")
        
        # 显示训练统计
        total_completed = len(framework.training_progress['completed_models'])
        total_failed = len(framework.training_progress['failed_models'])
        
        logger.info(f"\n[统计] 训练统计:")
        logger.info(f"   [成功] 完成: {total_completed} 个模型")
        logger.info(f"   ❌ 训练失败: {total_failed} 个模型")
        logger.info(f"   📁 模型保存位置: {framework.model_dir}")
        logger.info(f"   [保存] 结果位置: {framework.results_dir}")
        
        if total_failed > 0:
            logger.info(f"\n⚠️  失败的模型:")
            for failure in framework.training_progress['failed_models']:
                logger.info(f"   • {failure['model_name']}: {failure['error']}")
    else:
        logger.warning("⚠️  没有成功完成任何模型的训练")
    
        logger.info(f"\n[信息] 所有模型都使用了相同的训练策略，确保了实验的公平性")
    logger.info(f"🔄 支持断点续训 - 使用 --resume 参数可从断点继续")
    logger.info(f"🔧 强制重训 - 使用 --force_retrain 参数可重新训练指定模型")
    logger.info(f"📥 模型下载 - 使用 --download_models 参数可重新下载模型")
    logger.info(f"🌐 在线模式 - 使用 --use_online_models 参数可直接使用在线模型")
    
    # 显示本地模型状态
    if framework.use_local_models:
        logger.info(f"\n[本地模型] 状态信息:")
        downloaded_models = framework.local_model_manager.list_downloaded_models()
        if downloaded_models:
            logger.info(f"   ✅ 已下载 {len(downloaded_models)} 个模型:")
            for local_name, info in downloaded_models.items():
                model_path = Path(info['local_path'])
                logger.info(f"      • {info['hf_name']} → {model_path.name}")
        else:
            logger.info(f"   ⚠️ 无本地模型，训练时将使用在线模型")
    else:
        logger.info(f"\n[在线模式] 将直接从 Hugging Face 加载模型")


if __name__ == "__main__":
    main()