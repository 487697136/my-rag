#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批次LLM辅助问题复杂度标注工具
支持断点续传、自动保存、进度恢复功能
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
import requests
from dataclasses import dataclass
import openai
from openai import OpenAI
import sys
from tqdm import tqdm
from collections import Counter
import shutil
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_llm_labeling.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BatchLabelingConfig:
    """分批次标注配置"""
    api_key: str
    model: str = "qwen-turbo"
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 50  # 每批次处理的样本数
    temperature: float = 0.1
    max_tokens: int = 500
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Self-Consistency 配置
    enable_self_consistency: bool = True
    num_consistency_samples: int = 3
    consistency_temperature: float = 0.7
    consistency_threshold: float = 0.6
    mark_inconsistent_for_review: bool = True
    
    # 分批次处理配置
    enable_auto_backup: bool = True  # 是否启用自动备份
    backup_frequency: int = 5  # 每处理多少批次进行一次备份
    progress_file: str = "labeling_progress.json"  # 进度文件
    backup_dir: str = "backup"  # 备份目录

class BatchLLMLabelingTool:
    """分批次LLM辅助标注工具"""
    
    def __init__(self, config: BatchLabelingConfig):
        self.config = config
        if config.base_url:
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = OpenAI(api_key=config.api_key)
        
        # 创建备份目录
        if self.config.enable_auto_backup:
            os.makedirs(self.config.backup_dir, exist_ok=True)
        
        self.labeling_criteria = """
You are an expert in question complexity classification. Please classify the question into one of three categories based on these definitions:

Definitions:
- Zero-hop: A factual question that the model itself can answer confidently and accurately without retrieving any external knowledge or performing any reasoning across documents.
- One-hop: A question that the model cannot ensure correctness on its own; it requires retrieval of knowledge from at most one or two documents (i.e., single-hop or simple two-hop reasoning).
- Multi-hop: A question that requires retrieval from multiple documents and integration of information via multi-step reasoning (more than two hops).

Task:
1. Describe your step-by-step reasoning about whether the model can answer this question without external retrieval, or requires one-hop or multi-hop retrieval.
2. End with one of the exact labels: Label: zero_hop / one_hop / multi_hop

Examples:

Q: "What is the capital city of Brazil?"
A: This is a simple factual question; a model can answer it confidently from internal knowledge. → Label: zero_hop

Q: "Who is the author of the book '1984'?"
A: A model likely needs to retrieve a document (e.g., Wikipedia) to be certain of the exact height. Only one retrieval step. → Label: one_hop

Q: "Who wrote the preface of '1984'?"
A: A model likely needs to retrieve a document (e.g., Wikipedia) to be certain of the exact height. Only one retrieval step. → Label: one_hop

Q: "Compare the performance of BERT and GPT on sentiment analysis tasks."
A: This requires retrieving information about both BERT and GPT models from separate sources and synthesizing comparison—multi‑step reasoning. → Label: multi_hop

Guidelines:
- Provide detailed reasoning for your classification decision
- Focus on the number of retrieval steps and reasoning complexity
- Consider whether the model can answer confidently without external sources
"""
        
        self.labeling_prompt_template = """
Classify the following question:
Q: "{query}"

Answer (provide your reasoning, then final label):
"""
    
    def load_progress(self) -> Dict[str, Any]:
        """加载标注进度"""
        if os.path.exists(self.config.progress_file):
            try:
                with open(self.config.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"加载进度文件: 已完成 {progress.get('completed_batches', 0)} 个批次")
                return progress
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        
        return {
            "completed_batches": 0,
            "total_batches": 0,
            "completed_samples": 0,
            "total_samples": 0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "failed_samples": [],
            "batch_results": []
        }
    
    def save_progress(self, progress: Dict[str, Any]):
        """保存标注进度"""
        progress["last_update"] = datetime.now().isoformat()
        try:
            with open(self.config.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    def create_backup(self, output_file: str, batch_num: int):
        """创建备份文件"""
        if not self.config.enable_auto_backup:
            return
        
        try:
            backup_filename = f"backup_batch_{batch_num:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(self.config.backup_dir, backup_filename)
            
            if os.path.exists(output_file):
                shutil.copy2(output_file, backup_path)
                logger.info(f"创建备份: {backup_path}")
        except Exception as e:
            logger.warning(f"创建备份失败: {e}")
    
    def split_into_batches(self, data: List[Dict], batch_size: int) -> List[List[Dict]]:
        """将数据分割成批次"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def create_labeling_prompt(self, query: str) -> str:
        """创建标注提示"""
        return self.labeling_prompt_template.format(query=query)
    
    def parse_llm_response(self, response: str, query: str) -> dict:
        """解析LLM响应"""
        import re
        
        try:
            # 查找标签
            label_pattern = r'Label:\s*(zero_hop|one_hop|multi_hop)'
            label_match = re.search(label_pattern, response, re.IGNORECASE)
            
            if label_match:
                label = label_match.group(1).lower()
                reasoning = response[:label_match.start()].strip()
                
                return {
                    "query": query,
                    "label": label,
                    "reasoning": reasoning,
                    "raw_response": response
                }
            else:
                # 尝试其他模式
                if "zero_hop" in response.lower():
                    label = "zero_hop"
                elif "one_hop" in response.lower():
                    label = "one_hop"
                elif "multi_hop" in response.lower():
                    label = "multi_hop"
                else:
                    logger.warning(f"无法从响应中提取标签: {response[:100]}...")
                    return None
                
                return {
                    "query": query,
                    "label": label,
                    "reasoning": response,
                    "raw_response": response
                }
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}, 响应: {response[:200]}...")
            return None
    
    def call_llm_api_with_temperature(self, prompt: str, temperature: float) -> str:
        """使用指定温度调用LLM API"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.labeling_criteria},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"API调用最终失败: {e}")
                    return None
    
    def label_with_self_consistency(self, question: str) -> Optional[Dict[str, Any]]:
        """使用Self-Consistency策略标注单个问题"""
        prompt = self.create_labeling_prompt(question)
        responses = []
        labels = []
        
        # 多次采样推理
        for i in range(self.config.num_consistency_samples):
            try:
                response = self.call_llm_api_with_temperature(prompt, self.config.consistency_temperature)
                if response:
                    result = self.parse_llm_response(response, question)
                    if result:
                        responses.append(result)
                        labels.append(result.get("label", "UNKNOWN"))
                time.sleep(0.2)  # 避免API限流
            except Exception as e:
                logger.warning(f"采样{i+1}失败: {e}")
                continue
        
        if not labels:
            logger.error(f"所有采样都失败: {question}")
            return None
        
        # 投票机制确定最终标签
        return self.vote_for_final_label(labels, responses, question)
    
    def vote_for_final_label(self, labels: List[str], responses: List[Dict], question: str) -> Dict[str, Any]:
        """投票机制确定最终标签"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        most_common_label, max_count = label_counts.most_common(1)[0]
        consistency_ratio = max_count / total_samples
        
        # 判断是否达到一致性阈值
        if consistency_ratio >= self.config.consistency_threshold:
            final_label = most_common_label
            confidence = consistency_ratio
            
            selected_response = None
            for response in responses:
                if response.get("label") == most_common_label:
                    selected_response = response
                    break
            
            reasoning = selected_response.get("reasoning", "") if selected_response else ""
            raw_response = selected_response.get("raw_response", "") if selected_response else ""
            
        else:
            if self.config.mark_inconsistent_for_review:
                final_label = "to_review"
                confidence = consistency_ratio
                reasoning = f"自一致性不足({consistency_ratio:.2f} < {self.config.consistency_threshold})，标签分布: {dict(label_counts)}"
                raw_response = f"多次采样结果不一致: {[r.get('label') for r in responses]}"
            else:
                final_label = most_common_label
                confidence = consistency_ratio
                selected_response = None
                for response in responses:
                    if response.get("label") == most_common_label:
                        selected_response = response
                        break
                reasoning = selected_response.get("reasoning", "") if selected_response else ""
                raw_response = selected_response.get("raw_response", "") if selected_response else ""
        
        return {
            "query": question,
            "complexity": final_label,  # 直接使用complexity字段
            "reasoning": reasoning,
            "raw_response": raw_response,
            "consistency_info": {
                "samples_count": total_samples,
                "label_distribution": dict(label_counts),
                "consistency_ratio": consistency_ratio,
                "all_labels": labels,
                "final_confidence": confidence
            }
        }
    
    def label_single_question_basic(self, question: str) -> Optional[Dict[str, Any]]:
        """基础单次标注"""
        prompt = self.create_labeling_prompt(question)
        response = self.call_llm_api_with_temperature(prompt, self.config.temperature)
        
        if response is None:
            return None
        
        result = self.parse_llm_response(response, question)
        if result is None:
            return None
        
        return {
            "query": question,
            "complexity": result.get("label", "UNKNOWN"),  # 直接使用complexity字段
            "reasoning": result.get("reasoning", ""),
            "raw_response": result.get("raw_response", response)
        }
    
    def label_single_question(self, question: str) -> Optional[Dict[str, Any]]:
        """标注单个问题"""
        if self.config.enable_self_consistency:
            return self.label_with_self_consistency(question)
        else:
            return self.label_single_question_basic(question)
    
    def label_batch(self, batch: List[Dict[str, Any]], batch_num: int, total_batches: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        """标注单个批次"""
        results = []
        failed = []
        
        logger.info(f"开始处理批次 {batch_num}/{total_batches} (包含 {len(batch)} 个样本)")
        
        pbar = tqdm(total=len(batch), desc=f"批次 {batch_num}/{total_batches}", ncols=80)
        
        for item in batch:
            query = item["query"]
            
            # 跳过已标注的样本
            if item.get("complexity") and item["complexity"] != "null":
                logger.debug(f"跳过已标注样本: {query[:50]}...")
                results.append(item)
                pbar.update(1)
                continue
            
            result = self.label_single_question(query)
            
            if result and result.get("complexity") not in ["FAILED", "UNKNOWN"]:
                # 保留原始数据结构，只更新complexity字段
                updated_item = item.copy()
                updated_item["complexity"] = result["complexity"]
                results.append(updated_item)
                logger.debug(f"标注完成: {query[:50]}... -> {result['complexity']}")
            else:
                logger.warning(f"标注失败: {query}")
                failed.append(query)
                # 保留原始项目，但标记为失败
                failed_item = item.copy()
                failed_item["complexity"] = "FAILED"
                results.append(failed_item)
            
            pbar.update(1)
            # 减少API调用频率
            time.sleep(0.3)
        
        pbar.close()
        
        logger.info(f"批次 {batch_num} 完成: 成功 {len(results) - len(failed)}/{len(batch)} 个样本")
        return results, failed
    
    def merge_batch_results(self, original_data: List[Dict], batch_results: List[Dict], 
                          start_idx: int, end_idx: int) -> List[Dict]:
        """将批次结果合并到原始数据中"""
        merged_data = original_data.copy()
        
        for i, result in enumerate(batch_results):
            if start_idx + i < len(merged_data):
                merged_data[start_idx + i] = result
        
        return merged_data
    
    def save_incremental_results(self, data: List[Dict], output_file: str):
        """增量保存结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def print_batch_summary(self, batch_num: int, total_batches: int, 
                          batch_results: List[Dict], all_failed: List[str]):
        """打印批次摘要"""
        successful = len([r for r in batch_results if r.get("complexity") not in ["FAILED", "null"]])
        
        print(f"\n批次 {batch_num}/{total_batches} 完成:")
        print(f"  成功标注: {successful}/{len(batch_results)}")
        print(f"  累计失败: {len(all_failed)}")
        
        # 统计当前批次的标签分布
        batch_labels = [r.get("complexity") for r in batch_results if r.get("complexity") not in ["FAILED", "null"]]
        if batch_labels:
            label_dist = Counter(batch_labels)
            print(f"  本批次标签分布: {dict(label_dist)}")
    
    def batch_label_file(self, input_file: str, output_file: str, 
                        resume: bool = True) -> Dict[str, Any]:
        """分批次标注文件"""
        
        # 加载原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        total_samples = len(original_data)
        total_batches = math.ceil(total_samples / self.config.batch_size)
        
        # 加载进度
        progress = self.load_progress() if resume else {
            "completed_batches": 0,
            "total_batches": total_batches,
            "completed_samples": 0,
            "total_samples": total_samples,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "failed_samples": [],
            "batch_results": []
        }
        
        progress["total_batches"] = total_batches
        progress["total_samples"] = total_samples
        
        # 如果是新开始的任务，初始化输出文件
        if progress["completed_batches"] == 0:
            self.save_incremental_results(original_data, output_file)
        
        # 加载当前的工作数据
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                working_data = json.load(f)
        else:
            working_data = original_data.copy()
        
        all_failed = progress.get("failed_samples", [])
        start_batch = progress["completed_batches"]
        
        logger.info(f"开始分批次标注: 从批次 {start_batch + 1} 开始，共 {total_batches} 批次")
        
        try:
            # 分批次处理
            for batch_num in range(start_batch, total_batches):
                start_idx = batch_num * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, total_samples)
                
                current_batch = working_data[start_idx:end_idx]
                
                # 标注当前批次
                batch_results, batch_failed = self.label_batch(
                    current_batch, batch_num + 1, total_batches
                )
                
                # 更新工作数据
                working_data = self.merge_batch_results(
                    working_data, batch_results, start_idx, end_idx
                )
                
                # 累积失败样本
                all_failed.extend(batch_failed)
                
                # 保存结果
                self.save_incremental_results(working_data, output_file)
                
                # 更新进度
                progress["completed_batches"] = batch_num + 1
                progress["completed_samples"] = min(end_idx, total_samples)
                progress["failed_samples"] = all_failed
                self.save_progress(progress)
                
                # 打印批次摘要
                self.print_batch_summary(batch_num + 1, total_batches, batch_results, all_failed)
                
                # 创建备份
                if (batch_num + 1) % self.config.backup_frequency == 0:
                    self.create_backup(output_file, batch_num + 1)
                
                logger.info(f"批次 {batch_num + 1}/{total_batches} 完成，总进度: {progress['completed_samples']}/{total_samples}")
        
        except KeyboardInterrupt:
            logger.info("用户中断，正在保存当前进度...")
            self.save_progress(progress)
            self.save_incremental_results(working_data, output_file)
            print(f"\n进度已保存，下次可从批次 {progress['completed_batches'] + 1} 继续")
            return progress
        
        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            self.save_progress(progress)
            self.save_incremental_results(working_data, output_file)
            raise
        
        # 完成所有批次
        logger.info("所有批次标注完成!")
        
        # 生成最终统计
        final_stats = self.generate_final_statistics(working_data, all_failed, progress)
        
        # 创建最终备份
        self.create_backup(output_file, total_batches)
        
        return final_stats
    
    def generate_final_statistics(self, data: List[Dict], failed_samples: List[str], 
                                progress: Dict) -> Dict[str, Any]:
        """生成最终统计信息"""
        total_samples = len(data)
        successful_samples = len([item for item in data if item.get("complexity") not in ["FAILED", "null"]])
        failed_count = len(failed_samples)
        
        # 统计标签分布
        label_distribution = Counter()
        for item in data:
            complexity = item.get("complexity")
            if complexity and complexity not in ["FAILED", "null"]:
                label_distribution[complexity] += 1
        
        stats = {
            "summary": {
                "total_samples": total_samples,
                "successful_samples": successful_samples,
                "failed_samples": failed_count,
                "success_rate": successful_samples / total_samples if total_samples > 0 else 0,
                "label_distribution": dict(label_distribution)
            },
            "timing": {
                "start_time": progress.get("start_time"),
                "end_time": datetime.now().isoformat(),
                "total_batches": progress.get("total_batches", 0),
                "completed_batches": progress.get("completed_batches", 0)
            },
            "configuration": {
                "batch_size": self.config.batch_size,
                "model": self.config.model,
                "self_consistency_enabled": self.config.enable_self_consistency,
                "consistency_samples": self.config.num_consistency_samples if self.config.enable_self_consistency else None,
                "consistency_threshold": self.config.consistency_threshold if self.config.enable_self_consistency else None
            }
        }
        
        print("\n" + "="*60)
        print("最终标注统计")
        print("="*60)
        print(f"总样本数: {total_samples}")
        print(f"成功标注: {successful_samples} ({successful_samples/total_samples*100:.1f}%)")
        print(f"失败数量: {failed_count}")
        print(f"标签分布:")
        for label, count in label_distribution.most_common():
            print(f"  {label}: {count} ({count/successful_samples*100:.1f}%)")
        print("="*60)
        
        return stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分批次LLM辅助问题复杂度标注工具")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--api-key", default=None, help="API密钥 (可选，会自动从环境变量DASHSCOPE_API_KEY获取)")
    parser.add_argument("--model", default="qwen-turbo", help="使用的模型 (默认: qwen-turbo)")
    parser.add_argument("--temperature", type=float, default=0.1, help="温度参数 (默认: 0.1)")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                       help="API base_url (默认: 阿里云DashScope)")
    parser.add_argument("--batch-size", type=int, default=50, help="每批次处理的样本数 (默认: 50)")
    parser.add_argument("--disable-self-consistency", action="store_true", help="禁用自一致性策略，使用单次推理")
    parser.add_argument("--consistency-samples", type=int, default=3, help="自一致性采样次数 (默认: 3)")
    parser.add_argument("--consistency-threshold", type=float, default=0.6, help="一致性阈值 (默认: 0.6)")
    parser.add_argument("--backup-frequency", type=int, default=5, help="备份频率（每多少批次备份一次，默认: 5）")
    parser.add_argument("--disable-backup", action="store_true", help="禁用自动备份")
    parser.add_argument("--no-resume", action="store_true", help="不恢复之前的进度，重新开始")
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("请设置DASHSCOPE_API_KEY环境变量或使用--api-key参数")
            return
    
    # 创建配置
    config = BatchLabelingConfig(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        batch_size=args.batch_size,
        enable_self_consistency=not args.disable_self_consistency,
        num_consistency_samples=args.consistency_samples,
        consistency_threshold=args.consistency_threshold,
        consistency_temperature=args.temperature + 0.6,
        enable_auto_backup=not args.disable_backup,
        backup_frequency=args.backup_frequency
    )
    
    # 创建标注工具
    labeling_tool = BatchLLMLabelingTool(config)
    
    try:
        logger.info(f"开始分批次标注: {args.input} -> {args.output}")
        stats = labeling_tool.batch_label_file(
            args.input, 
            args.output, 
            resume=not args.no_resume
        )
        
        # 保存统计信息
        stats_file = args.output.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"标注完成！统计信息已保存到: {stats_file}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()