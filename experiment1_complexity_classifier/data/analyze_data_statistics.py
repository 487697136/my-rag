#!/usr/bin/env python3
"""
数据复杂度统计分析工具

该脚本用于分析标注数据文件夹中各复杂度问题的数量和占比。
遵循项目规范，使用核心库功能，提供详细的统计报告。

Author: AI Assistant
Date: 2025-01-07
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DataComplexityAnalyzer:
    """数据复杂度分析器"""
    
    def __init__(self, data_dir: str):
        """
        初始化分析器
        
        Args:
            data_dir: 标注数据文件夹路径
        """
        self.data_dir = Path(data_dir)
        self.complexity_stats = defaultdict(int)
        self.file_stats = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0
        
    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            解析后的数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ 成功加载文件: {file_path.name} ({len(data)} 条记录)")
            return data
        except Exception as e:
            print(f"✗ 加载文件失败: {file_path.name}, 错误: {e}")
            return []
    
    def analyze_single_file(self, file_path: Path) -> Dict[str, int]:
        """
        分析单个文件的复杂度分布
        
        Args:
            file_path: 文件路径
            
        Returns:
            复杂度统计字典
        """
        data = self.load_json_file(file_path)
        if not data:
            return {}
        
        file_complexity_count = Counter()
        
        for item in data:
            if 'complexity' in item:
                complexity = item['complexity']
                file_complexity_count[complexity] += 1
                self.complexity_stats[complexity] += 1
                self.file_stats[file_path.name][complexity] += 1
        
        self.total_samples += len(data)
        return dict(file_complexity_count)
    
    def analyze_all_files(self) -> None:
        """分析所有JSON文件"""
        print("=" * 60)
        print("开始分析标注数据文件夹...")
        print("=" * 60)
        
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print("未找到JSON文件！")
            return
        
        print(f"找到 {len(json_files)} 个JSON文件:")
        for file_path in json_files:
            print(f"  - {file_path.name}")
        
        print("\n开始逐文件分析...")
        print("-" * 40)
        
        for file_path in json_files:
            file_stats = self.analyze_single_file(file_path)
            if file_stats:
                print(f"文件 {file_path.name} 复杂度分布:")
                for complexity, count in sorted(file_stats.items()):
                    print(f"  {complexity}: {count} 条")
                print()
    
    def generate_summary_report(self) -> None:
        """生成汇总报告"""
        print("=" * 60)
        print("数据复杂度统计汇总报告")
        print("=" * 60)
        
        print(f"总样本数量: {self.total_samples:,}")
        print(f"复杂度类别数: {len(self.complexity_stats)}")
        print()
        
        print("整体复杂度分布:")
        print("-" * 40)
        
        # 按复杂度排序 (zero_hop, one_hop, multi_hop)
        complexity_order = ['zero_hop', 'one_hop', 'multi_hop']
        sorted_complexities = []
        
        # 先添加已知顺序的复杂度
        for complexity in complexity_order:
            if complexity in self.complexity_stats:
                sorted_complexities.append(complexity)
        
        # 再添加其他未知复杂度
        for complexity in sorted(self.complexity_stats.keys()):
            if complexity not in sorted_complexities:
                sorted_complexities.append(complexity)
        
        for complexity in sorted_complexities:
            count = self.complexity_stats[complexity]
            percentage = (count / self.total_samples) * 100
            print(f"{complexity:>12}: {count:>8,} 条 ({percentage:>6.2f}%)")
        
        print()
        print("各文件复杂度分布:")
        print("-" * 40)
        
        for filename, file_complexity_stats in self.file_stats.items():
            file_total = sum(file_complexity_stats.values())
            print(f"\n{filename} (总计: {file_total:,} 条):")
            
            for complexity in sorted_complexities:
                if complexity in file_complexity_stats:
                    count = file_complexity_stats[complexity]
                    percentage = (count / file_total) * 100
                    print(f"  {complexity:>12}: {count:>8,} 条 ({percentage:>6.2f}%)")
    
    def create_visualizations(self) -> None:
        """创建可视化图表"""
        if not self.complexity_stats:
            print("没有数据可供可视化")
            return
        
        print("\n生成可视化图表...")
        
        # 创建输出目录
        output_dir = self.data_dir.parent / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        # 准备数据
        complexity_order = ['zero_hop', 'one_hop', 'multi_hop']
        sorted_complexities = []
        
        for complexity in complexity_order:
            if complexity in self.complexity_stats:
                sorted_complexities.append(complexity)
        
        for complexity in sorted(self.complexity_stats.keys()):
            if complexity not in sorted_complexities:
                sorted_complexities.append(complexity)
        
        counts = [self.complexity_stats[c] for c in sorted_complexities]
        percentages = [(c / self.total_samples) * 100 for c in counts]
        
        # 图1: 整体复杂度分布柱状图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(sorted_complexities, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('问题复杂度分布 - 数量', fontsize=14, fontweight='bold')
        plt.xlabel('复杂度类型')
        plt.ylabel('样本数量')
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 图2: 复杂度分布饼图
        plt.subplot(2, 2, 2)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        wedges, texts, autotexts = plt.pie(counts, labels=sorted_complexities, autopct='%1.1f%%',
                                          colors=colors[:len(sorted_complexities)], startangle=90)
        plt.title('问题复杂度分布 - 占比', fontsize=14, fontweight='bold')
        
        # 图3: 各文件复杂度分布堆叠柱状图
        plt.subplot(2, 1, 2)
        
        # 准备文件数据
        file_data = []
        file_names = []
        
        for filename, file_complexity_stats in self.file_stats.items():
            file_names.append(filename.replace('.json', '').replace('_', '\n'))
            file_counts = [file_complexity_stats.get(c, 0) for c in sorted_complexities]
            file_data.append(file_counts)
        
        # 转换为numpy数组便于堆叠
        import numpy as np
        file_data = np.array(file_data).T
        
        # 创建堆叠柱状图
        bottom = np.zeros(len(file_names))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (complexity, color) in enumerate(zip(sorted_complexities, colors)):
            plt.bar(file_names, file_data[i], bottom=bottom, label=complexity, color=color)
            bottom += file_data[i]
        
        plt.title('各文件复杂度分布', fontsize=14, fontweight='bold')
        plt.xlabel('数据文件')
        plt.ylabel('样本数量')
        plt.legend(title='复杂度类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = output_dir / "complexity_distribution_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存: {chart_path}")
        
        plt.show()
    
    def save_detailed_report(self) -> None:
        """保存详细报告到文件"""
        output_dir = self.data_dir.parent / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        # 保存CSV格式的统计数据
        csv_path = output_dir / "complexity_statistics.csv"
        
        # 整体统计
        overall_data = []
        for complexity, count in self.complexity_stats.items():
            percentage = (count / self.total_samples) * 100
            overall_data.append({
                'complexity': complexity,
                'count': count,
                'percentage': percentage
            })
        
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 统计数据已保存: {csv_path}")
        
        # 保存详细报告
        report_path = output_dir / "detailed_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("数据复杂度统计分析详细报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.data_dir}\n")
            f.write(f"总样本数量: {self.total_samples:,}\n")
            f.write(f"复杂度类别数: {len(self.complexity_stats)}\n\n")
            
            f.write("整体复杂度分布:\n")
            f.write("-" * 40 + "\n")
            
            complexity_order = ['zero_hop', 'one_hop', 'multi_hop']
            sorted_complexities = []
            
            for complexity in complexity_order:
                if complexity in self.complexity_stats:
                    sorted_complexities.append(complexity)
            
            for complexity in sorted(self.complexity_stats.keys()):
                if complexity not in sorted_complexities:
                    sorted_complexities.append(complexity)
            
            for complexity in sorted_complexities:
                count = self.complexity_stats[complexity]
                percentage = (count / self.total_samples) * 100
                f.write(f"{complexity:>12}: {count:>8,} 条 ({percentage:>6.2f}%)\n")
            
            f.write("\n各文件复杂度分布:\n")
            f.write("-" * 40 + "\n")
            
            for filename, file_complexity_stats in self.file_stats.items():
                file_total = sum(file_complexity_stats.values())
                f.write(f"\n{filename} (总计: {file_total:,} 条):\n")
                
                for complexity in sorted_complexities:
                    if complexity in file_complexity_stats:
                        count = file_complexity_stats[complexity]
                        percentage = (count / file_total) * 100
                        f.write(f"  {complexity:>12}: {count:>8,} 条 ({percentage:>6.2f}%)\n")
        
        print(f"✓ 详细报告已保存: {report_path}")


def main():
    """主函数"""
    print("数据复杂度统计分析工具")
    print("=" * 60)
    
    # 设置数据目录
    data_dir = "data/labeled_data"
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 创建分析器并执行分析
    analyzer = DataComplexityAnalyzer(data_dir)
    
    try:
        # 分析所有文件
        analyzer.analyze_all_files()
        
        # 生成汇总报告
        analyzer.generate_summary_report()
        
        # 创建可视化图表
        analyzer.create_visualizations()
        
        # 保存详细报告
        analyzer.save_detailed_report()
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("结果文件保存在: data/analysis_results/")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()