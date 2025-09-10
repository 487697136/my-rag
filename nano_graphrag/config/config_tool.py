#!/usr/bin/env python3
"""
配置管理工具

提供命令行配置管理功能
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nano_graphrag.config import (
    load_config_template,
    load_config,
    save_config,
    create_config_from_template,
    validate_config,
    get_default_config
)

def create_config_command(args):
    """创建配置文件命令"""
    try:
        config = create_config_from_template(
            args.output,
            working_dir=args.working_dir,
            api_type=args.api_type,
            model_name=args.model_name
        )
        print(f"✅ 配置文件创建成功: {args.output}")
        return True
    except Exception as e:
        print(f"❌ 创建配置文件失败: {e}")
        return False

def validate_config_command(args):
    """验证配置文件命令"""
    try:
        config = load_config(args.config_path)
        if validate_config(config):
            print("✅ 配置文件验证通过")
            return True
        else:
            print("❌ 配置文件验证失败")
            return False
    except Exception as e:
        print(f"❌ 验证配置文件失败: {e}")
        return False

def show_template_command(args):
    """显示配置模板命令"""
    try:
        config = load_config_template()
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"❌ 加载配置模板失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="nano-graphrag 配置管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建配置文件命令
    create_parser = subparsers.add_parser('create', help='创建配置文件')
    create_parser.add_argument('output', help='输出配置文件路径')
    create_parser.add_argument('--working-dir', default='./nano_graphrag_cache', help='工作目录')
    create_parser.add_argument('--api-type', default='dashscope', help='API类型')
    create_parser.add_argument('--model-name', default='qwen-turbo', help='模型名称')
    create_parser.set_defaults(func=create_config_command)
    
    # 验证配置文件命令
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config_path', help='配置文件路径')
    validate_parser.set_defaults(func=validate_config_command)
    
    # 显示模板命令
    template_parser = subparsers.add_parser('template', help='显示配置模板')
    template_parser.set_defaults(func=show_template_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = args.func(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 