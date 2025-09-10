"""
社区报告生成器

提供社区报告的生成功能。
"""

import asyncio
import sys
from typing import Dict, Any
from ..base import BaseKVStorage, CommunitySchema, BaseGraphStorage
from .._utils import logger
from ..answer_generation.prompts import PROMPTS
from .community_packer import _pack_single_community_describe


def _community_report_json_to_str(parsed_output: Dict[str, Any]) -> str:
    """
    将社区报告JSON转换为字符串
    
    Args:
        parsed_output: 解析后的输出字典
        
    Returns:
        格式化的报告字符串
    """
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: Dict[str, Any]) -> str:
        if isinstance(finding, str):
            return finding
        return finding.get("summary", "")

    def finding_explanation(finding: Dict[str, Any]) -> str:
        if isinstance(finding, str):
            return ""
        return finding.get("explanation", "")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: Dict[str, Any],
) -> None:
    """
    生成社区报告
    
    Args:
        community_report_kv: 社区报告键值存储
        knwoledge_graph_inst: 知识图谱实例
        global_config: 全局配置
    """
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]
    # 获取LLM缓存实例
    llm_cache = global_config.get("llm_response_cache")

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: Dict[str, Any], already_reports: Dict[str, CommunitySchema]
    ) -> Dict[str, Any]:
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        # 尝试使用新格式的模板参数
        try:
            # 解析describe中的实体和关系信息
            import re
            
            # 提取实体部分
            entities_match = re.search(r'-----Entities-----\n```csv\n(.*?)\n```', describe, re.DOTALL)
            entities_text = entities_match.group(1) if entities_match else "No entities found"
            
            # 提取关系部分
            relationships_match = re.search(r'-----Relationships-----\n```csv\n(.*?)\n```', describe, re.DOTALL)
            relationships_text = relationships_match.group(1) if relationships_match else "No relationships found"
            
            # 尝试新格式
            prompt = community_report_prompt.format(
                entities=entities_text,
                relationships=relationships_text,
                claims="No claims available"
            )
        except KeyError:
            # 如果新格式失败，回退到原格式
            prompt = community_report_prompt.format(input_text=describe)
        
        # 添加重试和错误处理
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 显式传递缓存参数
                if llm_cache:
                    response = await use_llm_func(prompt, hashing_kv=llm_cache, **llm_extra_kwargs)
                else:
                    response = await use_llm_func(prompt, **llm_extra_kwargs)
                
                data = use_string_json_convert_func(response)
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"社区报告生成失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    logger.warning(f"社区报告生成失败，使用默认空报告")
                    # 创建一个最小的有效报告
                    data = {
                        "title": f"社区 {community.get('title', 'Unknown')}",
                        "summary": "无法生成报告摘要。",
                        "rating": 0.0,
                        "rating_explanation": "由于技术原因无法评估。",
                        "findings": [
                            {
                                "summary": "无法获取详细发现",
                                "explanation": "由于处理过程中出现错误，无法生成详细发现。"
                            }
                        ]
                    }
                else:
                    # 等待一段时间后重试
                    await asyncio.sleep(2 ** retry_count)  # 指数退避
        
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]

        # Windows 控制台在部分环境下为 GBK，无法打印如盲文点阵等字符，这里做安全降级
        try:
            progress_text = f"{now_ticks} Processed {already_processed} communities\r"
            # 若当前 stdout 编码不支持，则忽略无法编码的字符
            enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
            progress_text = progress_text.encode(enc, errors="ignore").decode(enc, errors="ignore")
        except Exception:
            # 兜底：使用 ASCII 纯文本
            progress_text = f". Processed {already_processed} communities\r"

        print(progress_text, end="", flush=True)
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    
    # 分批处理每个级别的社区
    for level in levels:
        level_community_items = [
            (k, v)
            for k, v in zip(community_keys, community_values)
            if v["level"] == level
        ]
        
        if not level_community_items:
            continue
            
        this_level_community_keys, this_level_community_values = zip(*level_community_items)
        
        # 分批处理，每批最多处理5个社区
        batch_size = 5
        for i in range(0, len(this_level_community_keys), batch_size):
            batch_keys = this_level_community_keys[i:i+batch_size]
            batch_values = this_level_community_values[i:i+batch_size]
            
            logger.info(f"处理级别 {level} 的社区批次 {i//batch_size + 1}/{(len(this_level_community_keys) + batch_size - 1)//batch_size}")
            
            # 处理当前批次
            batch_reports = await asyncio.gather(
                *[
                    _form_single_community_report(c, community_datas)
                    for c in batch_values
                ]
            )
            
            # 更新社区数据
            batch_data = {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    batch_keys,
                    batch_reports,
                    batch_values,
                )
            }
            community_datas.update(batch_data)
            
            # 每批次处理完成后保存一次数据
            await community_report_kv.upsert(batch_data)
            
            # 批次间休息一下，避免API过载
            await asyncio.sleep(1)
    
    # 清理进度条，安全打印换行
    try:
        end_text = "\n"
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
        end_text = end_text.encode(enc, errors="ignore").decode(enc, errors="ignore")
        print(end_text, end="")
    except Exception:
        print()
    # 最终保存所有数据（可能有些冗余，但确保数据完整性）
    await community_report_kv.upsert(community_datas) 