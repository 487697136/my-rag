"""
配置管理模块

提供配置文件的加载、验证和管理功能
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .._utils import logger

# 配置文件路径
CONFIG_TEMPLATE_PATH = Path(__file__).parent / "config_template.json"

def load_config_template() -> Dict[str, Any]:
    """
    加载配置模板
    
    Returns:
        Dict[str, Any]: 配置模板字典
    """
    try:
        with open(CONFIG_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件模板不存在: {CONFIG_TEMPLATE_PATH}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"配置文件模板格式错误: {e}")
        raise

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config (Dict[str, Any]): 配置字典
        config_path (str): 配置文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"成功保存配置文件: {config_path}")
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise

def create_config_from_template(output_path: str, **overrides) -> Dict[str, Any]:
    """
    从模板创建配置文件
    
    Args:
        output_path (str): 输出配置文件路径
        **overrides: 要覆盖的配置项
        
    Returns:
        Dict[str, Any]: 创建的配置字典
    """
    # 加载模板
    config = load_config_template()
    
    # 应用覆盖项
    for key, value in overrides.items():
        if key == "model_name":
            # 特殊处理 model_name，设置到 answer_generator_config 中
            config["answer_generator_config"]["model_name"] = value
        elif key == "working_dir":
            config["working_dir"] = value
        elif key == "api_type":
            config["api_type"] = value
        else:
            # 对于其他键，使用点分隔符处理嵌套结构
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    # 保存配置文件
    save_config(config, output_path)
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件
    
    Args:
        config (Dict[str, Any]): 配置字典
        
    Returns:
        bool: 验证是否通过
    """
    required_fields = [
        "working_dir",
        "api_type",
        "answer_generator_config",
        "retrieval_config",
        "fusion_config"
    ]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"配置文件缺少必需字段: {field}")
            return False
    
    # 验证 answer_generator_config
    if "model_name" not in config["answer_generator_config"]:
        logger.error("answer_generator_config 缺少 model_name 字段")
        return False
    
    # 验证 retrieval_config
    required_retrievers = ["naive", "bm25", "local", "global"]
    for retriever in required_retrievers:
        if retriever not in config["retrieval_config"]:
            logger.error(f"retrieval_config 缺少 {retriever} 配置")
            return False
    
    logger.info("配置文件验证通过")
    return True

def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return load_config_template()

# 导出主要函数
__all__ = [
    'load_config_template',
    'load_config', 
    'save_config',
    'create_config_from_template',
    'validate_config',
    'get_default_config'
] 