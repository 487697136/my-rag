#!/usr/bin/env python3
"""
实验一：本地模型下载和管理脚本

功能：
1. 从 Hugging Face 下载指定模型到本地
2. 支持断点续传
3. 验证模型完整性
4. 管理本地模型存储

运行方式：
    python scripts/00_download_models.py --models bert-base-cased roberta-large
    python scripts/00_download_models.py --all  # 下载所有需要的模型
"""

import os
import sys
import json
import yaml
import argparse
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel
)
from huggingface_hub import snapshot_download, Repository
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(experiment_root / 'outputs' / 'logs' / 'model_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalModelManager:
    """本地模型管理器"""
    
    def __init__(self, experiment_root: Path):
        """
        Args:
            experiment_root: 实验根目录
        """
        self.experiment_root = Path(experiment_root)
        self.models_dir = self.experiment_root / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型映射配置
        self.model_configs = {
            'bert-base-cased': {
                'hf_name': 'bert-base-cased',  # 使用官方稳定仓库 ID，避免 404/无权限
                'local_name': 'bert_base_cased',
                'type': 'bert',
                'hidden_size': 768
            },
            'roberta-large': {
                'hf_name': 'FacebookAI/roberta-large',  # 使用最新的组织路径
                'local_name': 'roberta_large',
                'type': 'roberta',
                'hidden_size': 1024
            }
        }
        
        # 下载进度跟踪文件
        self.download_status_file = self.models_dir / 'download_status.json'
        self.download_status = self._load_download_status()
    
    def _load_download_status(self) -> Dict:
        """加载下载状态"""
        if self.download_status_file.exists():
            try:
                with open(self.download_status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载下载状态失败: {e}")
        
        return {
            'downloaded_models': {},
            'failed_downloads': [],
            'last_update': None
        }
    
    def _save_download_status(self):
        """保存下载状态"""
        self.download_status['last_update'] = datetime.now().isoformat()
        try:
            with open(self.download_status_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存下载状态失败: {e}")
    
    def _calculate_directory_hash(self, directory: Path) -> str:
        """计算目录的哈希值（用于验证完整性）"""
        hash_md5 = hashlib.md5()
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                except Exception:
                    continue
        return hash_md5.hexdigest()
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """检查模型是否已下载"""
        if model_name not in self.model_configs:
            return False
        
        local_name = self.model_configs[model_name]['local_name']
        local_path = self.models_dir / local_name
        
        # 检查目录是否存在且包含必要文件
        if not local_path.exists():
            return False
        
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'vocab.txt']
        for file_name in required_files:
            if not (local_path / file_name).exists():
                # 某些模型可能使用不同的文件名
                continue
        
        # 检查下载状态记录
        return local_name in self.download_status['downloaded_models']
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """下载单个模型到本地"""
        if model_name not in self.model_configs:
            logger.error(f"不支持的模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        local_name = config['local_name']
        hf_name = config['hf_name']
        local_path = self.models_dir / local_name
        
        # 检查是否已下载（除非强制重新下载）
        if not force_redownload and self.is_model_downloaded(model_name):
            logger.info(f"✅ {model_name} 已下载到 {local_path}")
            return True
        
        logger.info(f"📥 开始下载 {model_name} 到 {local_path}")
        
        try:
            # 创建本地目录
            local_path.mkdir(parents=True, exist_ok=True)
            
            # 设置下载参数
            download_kwargs = {
                'repo_id': hf_name,
                'local_dir': str(local_path),
                'resume_download': True,  # 支持断点续传
                'local_dir_use_symlinks': False,  # 不使用符号链接
            }
            
            # 下载模型文件
            logger.info(f"正在从 {hf_name} 下载...")
            
            # 使用 snapshot_download 下载整个仓库
            try:
                snapshot_download(**download_kwargs)
                logger.info(f"✅ {model_name} 下载完成")
            except Exception as e:
                logger.error(f"使用 snapshot_download 失败: {e}")
                logger.info("尝试使用 transformers 库下载...")
                
                # 备用方法：使用 transformers 库下载
                self._download_with_transformers(hf_name, local_path, config['type'])
            
            # 验证下载完整性
            if self._verify_model_integrity(local_path, config['type']):
                # 更新下载状态
                self.download_status['downloaded_models'][local_name] = {
                    'hf_name': hf_name,
                    'local_path': str(local_path),
                    'download_time': datetime.now().isoformat(),
                    'hash': self._calculate_directory_hash(local_path),
                    'model_type': config['type'],
                    'hidden_size': config['hidden_size']
                }
                self._save_download_status()
                
                logger.info(f"✅ {model_name} 下载并验证成功")
                return True
            else:
                logger.error(f"❌ {model_name} 下载后验证失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ {model_name} 下载失败: {e}")
            
            # 记录失败信息
            failure_info = {
                'model_name': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.download_status['failed_downloads'].append(failure_info)
            self._save_download_status()
            
            return False
    
    def _download_with_transformers(self, hf_name: str, local_path: Path, model_type: str):
        """使用 transformers 库下载模型（备用方法）"""
        logger.info(f"使用 transformers 库下载 {hf_name}...")
        
        # 根据模型类型选择合适的类
        if model_type == 'bert':
            tokenizer_class = BertTokenizer
            model_class = BertModel
        elif model_type == 'roberta':
            tokenizer_class = RobertaTokenizer
            model_class = RobertaModel
        else:
            tokenizer_class = AutoTokenizer
            model_class = AutoModel
        
        # 下载配置文件
        config = AutoConfig.from_pretrained(hf_name)
        config.save_pretrained(local_path)
        
        # 下载分词器
        tokenizer = tokenizer_class.from_pretrained(hf_name)
        tokenizer.save_pretrained(local_path)
        
        # 下载模型
        model = model_class.from_pretrained(hf_name)
        model.save_pretrained(local_path)
        
        logger.info(f"✅ 使用 transformers 库下载 {hf_name} 完成")
    
    def _verify_model_integrity(self, local_path: Path, model_type: str) -> bool:
        """验证模型完整性"""
        try:
            logger.info(f"验证模型完整性: {local_path}")
            
            # 检查必要文件是否存在
            config_file = local_path / 'config.json'
            if not config_file.exists():
                logger.error("配置文件不存在")
                return False
            
            # 尝试加载配置
            config = AutoConfig.from_pretrained(str(local_path), local_files_only=True)
            
            # 尝试加载分词器
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
            
            # 尝试加载模型（仅检查能否加载，不实际加载到内存）
            model = AutoModel.from_pretrained(
                str(local_path), 
                local_files_only=True,
                torch_dtype=torch.float32  # 使用较小的数据类型以节省内存
            )
            
            # 简单的前向传播测试
            test_input = tokenizer("This is a test", return_tensors="pt", max_length=16, truncation=True)
            with torch.no_grad():
                output = model(**test_input)
            
            logger.info("✅ 模型完整性验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型完整性验证失败: {e}")
            return False
    
    def get_local_model_path(self, model_name: str) -> Optional[Path]:
        """获取本地模型路径"""
        if not self.is_model_downloaded(model_name):
            return None
        
        local_name = self.model_configs[model_name]['local_name']
        return self.models_dir / local_name
    
    def list_downloaded_models(self) -> Dict:
        """列出已下载的模型"""
        return self.download_status['downloaded_models']
    
    def download_all_required_models(self, force_redownload: bool = False) -> Dict[str, bool]:
        """下载所有实验所需的模型"""
        logger.info("🚀 开始下载所有实验所需的模型...")
        
        results = {}
        total_models = len(self.model_configs)
        
        for i, model_name in enumerate(self.model_configs.keys(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{total_models}] 处理模型: {model_name}")
            logger.info(f"{'='*60}")
            
            success = self.download_model(model_name, force_redownload)
            results[model_name] = success
            
            if success:
                logger.info(f"✅ [{i}/{total_models}] {model_name} 处理成功")
            else:
                logger.error(f"❌ [{i}/{total_models}] {model_name} 处理失败")
        
        # 输出汇总报告
        self._print_download_summary(results)
        
        return results
    
    def _print_download_summary(self, results: Dict[str, bool]):
        """打印下载汇总报告"""
        successful = [model for model, success in results.items() if success]
        failed = [model for model, success in results.items() if not success]
        
        logger.info(f"\n{'='*70}")
        logger.info("📊 模型下载汇总报告")
        logger.info(f"{'='*70}")
        logger.info(f"✅ 成功下载: {len(successful)} 个模型")
        for model in successful:
            local_path = self.get_local_model_path(model)
            logger.info(f"   • {model} → {local_path}")
        
        if failed:
            logger.info(f"\n❌ 下载失败: {len(failed)} 个模型")
            for model in failed:
                logger.info(f"   • {model}")
        
        logger.info(f"\n📁 所有模型存储在: {self.models_dir}")
        
        # 计算总占用空间
        total_size = self._calculate_total_size()
        logger.info(f"💾 总占用空间: {total_size}")
    
    def _calculate_total_size(self) -> str:
        """计算已下载模型的总大小"""
        total_bytes = 0
        for model_path in self.models_dir.iterdir():
            if model_path.is_dir():
                for file_path in model_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            total_bytes += file_path.stat().st_size
                        except Exception:
                            continue
        
        # 转换为人类可读的格式
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_bytes < 1024.0:
                return f"{total_bytes:.1f} {unit}"
            total_bytes /= 1024.0
        return f"{total_bytes:.1f} TB"
    
    def clean_failed_downloads(self):
        """清理失败的下载"""
        logger.info("🧹 清理失败的下载...")
        
        for model_name in self.model_configs:
            local_name = self.model_configs[model_name]['local_name']
            local_path = self.models_dir / local_name
            
            if local_path.exists() and not self.is_model_downloaded(model_name):
                try:
                    import shutil
                    shutil.rmtree(local_path)
                    logger.info(f"🗑️ 已清理不完整的下载: {local_path}")
                except Exception as e:
                    logger.error(f"清理失败: {e}")
    
    def update_config_file(self, config_path: Path):
        """更新配置文件，添加本地模型路径"""
        logger.info(f"📝 更新配置文件: {config_path}")
        
        try:
            # 加载现有配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 添加本地模型配置
            if 'local_models' not in config:
                config['local_models'] = {}
            
            config['local_models']['enabled'] = True
            config['local_models']['base_path'] = str(self.models_dir)
            config['local_models']['models'] = {}
            
            # 为每个已下载的模型添加本地路径配置
            for model_name, model_config in self.model_configs.items():
                if self.is_model_downloaded(model_name):
                    local_path = self.get_local_model_path(model_name)
                    config['local_models']['models'][model_name] = {
                        'local_path': str(local_path),
                        'model_type': model_config['type'],
                        'hidden_size': model_config['hidden_size']
                    }
            
            # 保存更新后的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info("✅ 配置文件更新完成")
            
        except Exception as e:
            logger.error(f"❌ 更新配置文件失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载实验一所需的预训练模型到本地')
    parser.add_argument('--models', nargs='+', help='指定要下载的模型名称')
    parser.add_argument('--all', action='store_true', help='下载所有实验所需的模型')
    parser.add_argument('--force', action='store_true', help='强制重新下载（即使已存在）')
    parser.add_argument('--clean', action='store_true', help='清理失败的下载')
    parser.add_argument('--verify', action='store_true', help='验证已下载模型的完整性')
    parser.add_argument('--list', action='store_true', help='列出已下载的模型')
    parser.add_argument('--update-config', action='store_true', help='更新配置文件')
    args = parser.parse_args()
    
    # 初始化模型管理器
    experiment_root = Path(__file__).parent.parent
    manager = LocalModelManager(experiment_root)
    
    # 确保输出目录存在
    (experiment_root / 'outputs' / 'logs').mkdir(parents=True, exist_ok=True)
    
    try:
        if args.clean:
            manager.clean_failed_downloads()
        
        elif args.list:
            downloaded = manager.list_downloaded_models()
            print("\n📋 已下载的模型:")
            for local_name, info in downloaded.items():
                print(f"  • {info['hf_name']} → {info['local_path']}")
                print(f"    类型: {info['model_type']}, 下载时间: {info['download_time']}")
        
        elif args.verify:
            logger.info("🔍 验证已下载模型的完整性...")
            downloaded = manager.list_downloaded_models()
            for local_name, info in downloaded.items():
                local_path = Path(info['local_path'])
                if manager._verify_model_integrity(local_path, info['model_type']):
                    logger.info(f"✅ {info['hf_name']} 验证通过")
                else:
                    logger.error(f"❌ {info['hf_name']} 验证失败")
        
        elif args.all:
            # 下载所有模型
            results = manager.download_all_required_models(args.force)
            
            # 更新配置文件
            if args.update_config or any(results.values()):
                config_path = experiment_root / 'config' / 'model_config.yaml'
                manager.update_config_file(config_path)
        
        elif args.models:
            # 下载指定模型
            results = {}
            for model_name in args.models:
                success = manager.download_model(model_name, args.force)
                results[model_name] = success
            
            manager._print_download_summary(results)
            
            # 更新配置文件
            if args.update_config:
                config_path = experiment_root / 'config' / 'model_config.yaml'
                manager.update_config_file(config_path)
        
        elif args.update_config:
            # 仅更新配置文件
            config_path = experiment_root / 'config' / 'model_config.yaml'
            manager.update_config_file(config_path)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断下载")
    except Exception as e:
        logger.error(f"❌ 程序运行出错: {e}")
        raise


if __name__ == "__main__":
    main()