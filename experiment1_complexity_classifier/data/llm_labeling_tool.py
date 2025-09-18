#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM辅助问题复杂度标注工具
使用云服务LLM对问题进行zero_hop、one_hop、multi_hop分类标注
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import requests
from dataclasses import dataclass
import openai
from openai import OpenAI
import sys
from tqdm import tqdm
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_labeling.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LabelingConfig:
    """标注配置"""
    api_key: str
    model: str = "gpt-4"
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    temperature: float = 0.1
    max_tokens: int = 500
    base_url: str = None
    
    # Self-Consistency 配置
    enable_self_consistency: bool = True
    num_consistency_samples: int = 3  # 自一致性采样次数 (3-5次)
    consistency_temperature: float = 0.7  # 自一致性采样温度 (略高于标准温度)
    consistency_threshold: float = 0.6  # 一致性阈值 (>=60%才接受结果)
    mark_inconsistent_for_review: bool = True  # 是否将不一致的样本标记为待审核

class LLMLabelingTool:
    """LLM辅助标注工具"""
    
    def __init__(self, config: LabelingConfig):
        self.config = config
        if config.base_url:
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = OpenAI(api_key=config.api_key)
        
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
    
    def create_labeling_prompt(self, query: str) -> str:
        # 为单个问题创建提示
        return self.labeling_prompt_template.format(query=query)
    
    def parse_llm_response(self, response: str, query: str) -> dict:
        """解析单个问题的LLM响应"""
        import re
        
        try:
            # 查找标签
            label_pattern = r'Label:\s*(zero_hop|one_hop|multi_hop)'
            label_match = re.search(label_pattern, response, re.IGNORECASE)
            
            if label_match:
                label = label_match.group(1).lower()
                
                # 提取推理过程（标签前的内容）
                reasoning = response[:label_match.start()].strip()
                
                return {
                    "query": query,
                    "label": label,
                    "reasoning": reasoning,
                    "raw_response": response
                }
            else:
                # 如果没有找到正确的标签格式，尝试其他模式
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
    
    def call_llm_api(self, prompt: str) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.labeling_criteria},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.choices[0].message.content.strip()
                
                # 验证响应是否包含完整的JSON结构
                if '[' in result and ']' in result:
                    start = result.find('[')
                    end = result.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        return result
                    else:
                        logger.warning(f"响应JSON不完整，尝试重试 (尝试 {attempt + 1}/{self.config.max_retries})")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                else:
                    logger.warning(f"响应格式异常，尝试重试 (尝试 {attempt + 1}/{self.config.max_retries})")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                
                return result
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"API调用最终失败: {e}")
                    return None
    
    def label_single_question(self, question: str) -> Optional[Dict[str, Any]]:
        """标注单个问题"""
        if self.config.enable_self_consistency:
            return self.label_with_self_consistency(question)
        else:
            return self.label_single_question_basic(question)
    
    def label_single_question_basic(self, question: str) -> Optional[Dict[str, Any]]:
        """基础单次标注"""
        prompt = self.create_labeling_prompt(question)
        response = self.call_llm_api(prompt)
        
        if response is None:
            return None
        
        result = self.parse_llm_response(response, question)
        if result is None:
            return None
        
        return {
            "question": question,
            "label": result.get("label", "UNKNOWN"),
            "reasoning": result.get("reasoning", ""),
            "raw_response": result.get("raw_response", response)
        }
    
    def label_with_self_consistency(self, question: str) -> Optional[Dict[str, Any]]:
        """使用Self-Consistency策略标注单个问题"""
        logger.info(f"对问题进行自一致性标注 (采样{self.config.num_consistency_samples}次): {question[:50]}...")
        
        prompt = self.create_labeling_prompt(question)
        responses = []
        labels = []
        
        # 多次采样推理
        for i in range(self.config.num_consistency_samples):
            try:
                # 使用略高的温度进行采样
                response = self.call_llm_api_with_temperature(prompt, self.config.consistency_temperature)
                if response:
                    result = self.parse_llm_response(response, question)
                    if result:
                        responses.append(result)
                        labels.append(result.get("label", "UNKNOWN"))
                        logger.debug(f"采样{i+1}: {result.get('label', 'UNKNOWN')}")
                time.sleep(0.2)  # 避免API限流
            except Exception as e:
                logger.warning(f"采样{i+1}失败: {e}")
                continue
        
        if not labels:
            logger.error(f"所有采样都失败: {question}")
            return None
        
        # 投票机制确定最终标签
        final_result = self.vote_for_final_label(labels, responses, question)
        return final_result
    
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
    
    def vote_for_final_label(self, labels: List[str], responses: List[Dict], question: str) -> Dict[str, Any]:
        """投票机制确定最终标签"""
        
        # 统计标签频次
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # 获取最高票数标签
        most_common_label, max_count = label_counts.most_common(1)[0]
        consistency_ratio = max_count / total_samples
        
        logger.info(f"标签分布: {dict(label_counts)}, 最高票: {most_common_label}({max_count}/{total_samples}={consistency_ratio:.2f})")
        
        # 判断是否达到一致性阈值
        if consistency_ratio >= self.config.consistency_threshold:
            # 一致性足够，使用最高票标签
            final_label = most_common_label
            confidence = consistency_ratio
            
            # 选择对应的推理过程（优选最高票标签的第一个结果）
            selected_response = None
            for response in responses:
                if response.get("label") == most_common_label:
                    selected_response = response
                    break
            
            reasoning = selected_response.get("reasoning", "") if selected_response else ""
            raw_response = selected_response.get("raw_response", "") if selected_response else ""
            
        else:
            # 一致性不足
            if self.config.mark_inconsistent_for_review:
                final_label = "to_review"
                confidence = consistency_ratio
                reasoning = f"自一致性不足({consistency_ratio:.2f} < {self.config.consistency_threshold})，标签分布: {dict(label_counts)}"
                raw_response = f"多次采样结果不一致: {[r.get('label') for r in responses]}"
                logger.warning(f"标注不一致，标记为待审核: {question[:50]}...")
            else:
                # 仍然使用最高票标签，但降低置信度
                final_label = most_common_label
                confidence = consistency_ratio
                selected_response = None
                for response in responses:
                    if response.get("label") == most_common_label:
                        selected_response = response
                        break
                reasoning = selected_response.get("reasoning", "") if selected_response else ""
                raw_response = selected_response.get("raw_response", "") if selected_response else ""
                logger.warning(f"标注一致性较低，但仍使用最高票标签: {question[:50]}...")
        
        return {
            "question": question,
            "label": final_label,
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
    
    def label_batch_questions(self, questions: list) -> list:
        """批量标注问题"""
        results = []
        failed = []
        total_questions = len(questions)
        pbar = tqdm(total=total_questions, desc="LLM标注进度", ncols=80)
        for i, question in enumerate(questions, 1):
            result = self.label_single_question(question)
            if result:
                results.append(result)
                logger.info(f"标注: {result.get('question')} | label: {result.get('label')}")
            else:
                logger.warning(f"标注失败: {question}")
                failed.append(question)
                results.append({
                    "question": question,
                    "label": "FAILED",
                    "reasoning": "",
                    "raw_response": ""
                })
            pbar.update(1)
            # 减少API调用频率
            time.sleep(0.5)
        pbar.close()
        if failed:
            with open("failed_samples.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            logger.warning(f"有{len(failed)}条样本标注失败，已导出到failed_samples.json")
        logger.info(f"批量标注完成，成功标注 {len([r for r in results if r['label'] != 'FAILED'])}/{total_questions} 个问题")
        return results
    
    def load_questions_from_json(self, file_path: str) -> List[str]:
        """从JSON文件加载问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持多种JSON格式
            if isinstance(data, list):
                # 格式1: [{"question": "..."}, ...]
                if data and isinstance(data[0], dict) and "question" in data[0]:
                    return [item["question"] for item in data]
                # 格式2: ["question1", "question2", ...]
                elif data and isinstance(data[0], str):
                    return data
            elif isinstance(data, dict):
                # 格式3: {"questions": ["question1", "question2", ...]}
                if "questions" in data:
                    return data["questions"]
                # 格式4: {"data": [{"question": "..."}, ...]}
                elif "data" in data and isinstance(data["data"], list):
                    return [item["question"] for item in data["data"] if "question" in item]
            
            raise ValueError("不支持的JSON格式")
            
        except Exception as e:
            logger.error(f"加载问题文件失败: {e}")
            raise
    
    def save_results_to_json(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果到JSON文件"""
        try:
            # 创建输出目录（如果路径包含目录）
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 统计自一致性相关信息
            consistency_stats = self.get_consistency_statistics(results)
            
            # 准备输出数据
            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": len(results),
                    "successful_labels": len([r for r in results if r['label'] != 'FAILED']),
                    "failed_labels": len([r for r in results if r['label'] == 'FAILED']),
                    "to_review_labels": len([r for r in results if r['label'] == 'to_review']),
                    "model": self.config.model,
                    "label_distribution": self.get_label_distribution(results),
                    "self_consistency_enabled": self.config.enable_self_consistency,
                    "consistency_statistics": consistency_stats
                },
                "results": results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def get_consistency_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取自一致性统计信息"""
        if not self.config.enable_self_consistency:
            return {"enabled": False}
        
        consistency_ratios = []
        samples_counts = []
        
        for result in results:
            if "consistency_info" in result:
                consistency_ratios.append(result["consistency_info"]["consistency_ratio"])
                samples_counts.append(result["consistency_info"]["samples_count"])
        
        if consistency_ratios:
            avg_consistency = sum(consistency_ratios) / len(consistency_ratios)
            min_consistency = min(consistency_ratios)
            max_consistency = max(consistency_ratios)
            high_consistency_count = len([r for r in consistency_ratios if r >= self.config.consistency_threshold])
        else:
            avg_consistency = min_consistency = max_consistency = high_consistency_count = 0
        
        return {
            "enabled": True,
            "num_samples_per_question": self.config.num_consistency_samples,
            "consistency_threshold": self.config.consistency_threshold,
            "average_consistency_ratio": avg_consistency,
            "min_consistency_ratio": min_consistency,
            "max_consistency_ratio": max_consistency,
            "high_consistency_count": high_consistency_count,
            "total_evaluated": len(consistency_ratios)
        }
    
    def get_label_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取标注分布"""
        distribution = {}
        for result in results:
            label = result.get('label', 'unknown')
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印标注摘要"""
        if not results:
            print("没有标注结果")
            return
        
        distribution = self.get_label_distribution(results)
        total = len(results)
        consistency_stats = self.get_consistency_statistics(results)
        
        print("="*60)
        print("标注结果摘要")
        print("="*60)
        print(f"总样本数: {total}")
        print(f"标注分布:")
        for label, count in sorted(distribution.items()):
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # 打印自一致性统计
        if consistency_stats.get("enabled", False):
            print(f"\n自一致性统计:")
            print(f"  采样次数/问题: {consistency_stats['num_samples_per_question']}")
            print(f"  一致性阈值: {consistency_stats['consistency_threshold']:.2f}")
            print(f"  平均一致性: {consistency_stats['average_consistency_ratio']:.3f}")
            print(f"  一致性范围: {consistency_stats['min_consistency_ratio']:.3f} - {consistency_stats['max_consistency_ratio']:.3f}")
            print(f"  高一致性样本: {consistency_stats['high_consistency_count']}/{consistency_stats['total_evaluated']} "
                  f"({consistency_stats['high_consistency_count']/max(consistency_stats['total_evaluated'], 1)*100:.1f}%)")
        
        print("="*60)

    def overwrite_existing_labels(self, data: list) -> list:
        """覆盖已有标签，使用新的单个问题处理方式"""
        results = []
        failed = []
        pbar = tqdm(total=len(data), desc="LLM重新标注进度", ncols=80)
        
        for item in data:
            query = item["query"]
            result = self.label_single_question(query)
            
            if result and result.get("label") != "FAILED":
                results.append({
                    "query": query,
                    "label": result["label"],
                    "reasoning": result.get("reasoning", ""),
                    "raw_response": result.get("raw_response", "")
                })
                logger.info(f"重新标注: {query} | label: {result['label']}")
                    else:
                logger.warning(f"重新标注失败: {query}")
                failed.append(query)
            
            pbar.update(1)
            # 减少API调用频率
            time.sleep(0.5)
        
        pbar.close()
        
        # 只更新complexity字段，保留原有explanation
        query2label = {item["query"]: item["label"] for item in results}
        new_data = []
        for item in data:
            q = item["query"]
            if q in query2label:
                item["complexity"] = query2label[q]
                # 保留原有explanation，不修改
            new_data.append(item)
        
        # 导出失败样本
        if failed:
            with open("failed_samples.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            logger.warning(f"有{len(failed)}条样本标注失败，已导出到failed_samples.json")
        
        return new_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM辅助问题复杂度标注工具")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--api-key", default=None, help="API密钥 (可选，会自动从环境变量DASHSCOPE_API_KEY获取)")
    parser.add_argument("--model", default="qwen-turbo", help="使用的模型 (默认: qwen-turbo)")
    parser.add_argument("--temperature", type=float, default=0.1, help="温度参数 (默认: 0.1)")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                       help="API base_url (默认: 阿里云DashScope)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已标注数据，仅更新complexity字段")
    parser.add_argument("--disable-self-consistency", action="store_true", help="禁用自一致性策略，使用单次推理")
    parser.add_argument("--consistency-samples", type=int, default=3, help="自一致性采样次数 (默认: 3)")
    parser.add_argument("--consistency-threshold", type=float, default=0.6, help="一致性阈值 (默认: 0.6)")
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("请设置DASHSCOPE_API_KEY环境变量或使用--api-key参数")
            return
    
    # 创建配置
    config = LabelingConfig(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        enable_self_consistency=not args.disable_self_consistency,
        num_consistency_samples=args.consistency_samples,
        consistency_threshold=args.consistency_threshold,
        consistency_temperature=args.temperature + 0.6  # 自一致性采样温度略高
    )
    
    # 创建标注工具
    labeling_tool = LLMLabelingTool(config)
    
    try:
        # 加载问题
        logger.info(f"加载问题文件: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.overwrite:
            logger.info("覆盖标注模式：仅更新complexity字段")
            results = labeling_tool.overwrite_existing_labels(data)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存到: {args.output}")
        else:
            # 原有流程...
            questions = labeling_tool.load_questions_from_json(args.input)
            results = labeling_tool.label_batch_questions(questions)
            labeling_tool.save_results_to_json(results, args.output)
            labeling_tool.print_summary(results)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 