from typing import Union
import pickle
import asyncio
from openai import BadRequestError
from collections import defaultdict
import os
from typing import Optional
from openai import AsyncOpenAI
import dspy
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from nano_graphrag.answer_generation.prompts import PROMPTS
from nano_graphrag._utils import logger, compute_mdhash_id
from nano_graphrag.entity_extraction.module import TypedEntityRelationshipExtractor
# 这些函数需要重新实现或从其他模块导入
# from nano_graphrag._op import _merge_edges_then_upsert, _merge_nodes_then_upsert

# # 临时实现这些函数
# async def _merge_nodes_then_upsert(
#     nodes: list,
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict = {},
# ) -> None:
#     """临时实现：合并节点并更新到图存储"""
#     if not nodes:
#         return
    
#     # 批量添加节点
#     await knwoledge_graph_inst.add_nodes_batch(nodes)

# async def _merge_edges_then_upsert(
#     source: str,
#     target: str,
#     edges: list,
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict = {},
# ) -> None:
#     """临时实现：合并边并更新到图存储"""
#     if not edges:
#         return
    
#     # 批量添加边
#     await knwoledge_graph_inst.add_edges_batch(edges)

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """合并节点数据并更新到图数据库"""
    from collections import Counter
    from nano_graphrag._utils import split_string_by_multi_markers
    from nano_graphrag.answer_generation.prompts import GRAPH_FIELD_SEP
    
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    node_data = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(entity_name, node_data)
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """合并边数据并更新到图数据库"""
    from nano_graphrag._utils import split_string_by_multi_markers
    from nano_graphrag.answer_generation.prompts import GRAPH_FIELD_SEP
    
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True,
    global_config: dict = {},
) -> list[dspy.Example]:
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(
        chunk_key_dp: tuple[str, TextChunkSchema]
    ) -> dspy.Example:
        nonlocal already_processed, already_entities, already_relations
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        example = dspy.Example(
            input_text=content, entities=entities, relationships=relationships
        ).with_inputs("input_text")
        already_entities += len(entities)
        already_relations += len(relationships)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    filtered_examples = [
        example
        for example in examples
        if len(example.entities) > 0 and len(example.relationships) > 0
    ]
    num_filtered_examples = len(examples) - len(filtered_examples)
    if save_dataset:
        with open(filepath, "wb") as f:
            pickle.dump(filtered_examples, f)
            logger.info(
                f"Saved {len(filtered_examples)} examples with keys: {filtered_examples[0].keys()}, filtered {num_filtered_examples} examples"
            )

    return filtered_examples


async def extract_entities_dspy(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    using_amazon_bedrock: bool = False,  # 添加此参数以保持兼容性
) -> Union[BaseGraphStorage, None]:
    # 采用“纯文本 JSON 输出 + 解析”的抽取器接口（默认保留，可在非DashScope时使用）
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=False)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0
    total = len(ordered_chunks)

    # 实时进度显示辅助：每 N 条刷新一次
    progress_refresh_every = 1

    # 基于项目工具的并发限制装饰器（降低到 4 并发以减少网络压力）
    from nano_graphrag._utils import limit_async_func_call
    limiter = limit_async_func_call(4)

    async def _dashscope_complete_json(prompt: str, system_prompt: Optional[str] = None) -> str:
        """使用 DashScope OpenAI 兼容接口进行对话补全，返回文本。

        - 不使用 response_format，避免 JSON mode 要求。
        - 仅在检测到 DASHSCOPE_API_KEY 存在时调用。
        - 增加重试机制和错误处理。
        """
        import asyncio
        import random
        
        base_url = os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置，无法调用 DashScope。")
        
        # 配置客户端，增加超时设置
        client = AsyncOpenAI(
            base_url=base_url, 
            api_key=api_key,
            timeout=60.0  # 增加超时时间到60秒
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 实现指数退避重试机制
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model="qwen-turbo", 
                    messages=messages, 
                    temperature=0.1, 
                    max_tokens=8192
                )
                
                content = resp.choices[0].message.content or ""
                if content.strip():  # 确保返回了有效内容
                    return content
                else:
                    raise ValueError("API返回内容为空")
                    
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"DashScope API调用失败，已重试{max_retries}次: {str(e)}")
                    raise
                
                # 计算退避延迟
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # 添加随机抖动
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                logger.info(f"等待 {delay:.1f} 秒后重试...")
                await asyncio.sleep(delay)

    def _build_extraction_prompt(text: str) -> str:
        """构造抽取提示词，要求返回严格 JSON（字符串）。"""
        return (
            "你是信息抽取助手。请从输入文本中抽取实体与关系，严格输出一个 JSON 对象，"
            "仅包含 keys: entities, relationships。不要输出任何解释或Markdown。\n"
            "entities: 数组，每项包含 {entity_name, entity_type, description, importance_score}。\n"
            "relationships: 数组，每项包含 {src_id, tgt_id, description, weight, order}。\n"
            "约束: entity_type 只能取预定义枚举之一；importance_score/weight 为 0-1 浮点；order 为 1/2/3；"
            "src_id/tgt_id 必须精确匹配 entities.entity_name。\n\n"
            f"[TEXT]\n{text}"
        )

    @limiter
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            entities, relationships = [], []
            # 优先使用 DashScope 直连，彻底避免 response_format 注入
            if os.environ.get("DASHSCOPE_API_KEY"):
                prompt = _build_extraction_prompt(content)
                response_text = await _dashscope_complete_json(prompt)
                from nano_graphrag._utils import convert_response_to_json
                parsed = convert_response_to_json(response_text)
                raw_entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
                raw_relationships = parsed.get("relationships", []) if isinstance(parsed, dict) else []
                # 使用与模块一致的严格校验，保证后续字段完整
                from nano_graphrag.entity_extraction.module import Entity as _Entity, Relationship as _Relationship
                from pydantic import ValidationError as _ValidationError
                validated_entities = []
                for item in raw_entities:
                    if not isinstance(item, dict):
                        continue
                    try:
                        e = _Entity(**item)
                        validated_entities.append(e.to_dict())
                    except _ValidationError:
                        continue
                upper_names = {e["entity_name"].upper() for e in validated_entities}
                validated_relationships = []
                for item in raw_relationships:
                    if not isinstance(item, dict):
                        continue
                    try:
                        r = _Relationship(**item)
                        if r.src_id.upper() in upper_names and r.tgt_id.upper() in upper_names:
                            validated_relationships.append(r.to_dict())
                    except _ValidationError:
                        continue
                entities = validated_entities
                relationships = validated_relationships
            else:
                # 回退到 DSPy 非结构化抽取器（仍然不会启用 JSON mode）
                prediction = await asyncio.to_thread(entity_extractor, input_text=content)
                entities, relationships = prediction.entities, prediction.relationships
            logger.debug(f"成功提取 {len(entities)} 个实体和 {len(relationships)} 个关系")
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        except Exception as e:
            logger.error(f"意外错误在实体提取中: {e}")
            logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            entities, relationships = [], []

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for entity in entities:
            entity["source_id"] = chunk_key
            maybe_nodes[entity["entity_name"]].append(entity)
            already_entities += 1

        for relationship in relationships:
            relationship["source_id"] = chunk_key
            maybe_edges[(relationship["src_id"], relationship["tgt_id"])].append(
                relationship
            )
            already_relations += 1

        already_processed += 1
        if already_processed % progress_refresh_every == 0:
            # 使用ASCII安全的进度显示，避免GBK编码错误
            ascii_tick = "." if (already_processed % 2 == 1) else "-"
            progress_line = (
                f"{ascii_tick} 抽取进度 {already_processed}/{total} | "
                f"实体(含重复)={already_entities} | 关系(含重复)={already_relations}"
            )
            try:
                print(progress_line + "\r", end="", flush=True)
            except UnicodeEncodeError:
                # 退化为纯ASCII
                safe_line = (
                    f"{ascii_tick} Progress {already_processed}/{total} | "
                    f"Entities={already_entities} | Rels={already_relations}"
                )
                print(safe_line + "\r", end="", flush=True)
        return dict(maybe_nodes), dict(maybe_edges)

    # 如果提供了仅小样本验证的开关，从全量中截取前 50 条样本
    small_batch_mode = global_config.get("__small_batch_mode__", False)
    sample_size = global_config.get("__small_batch_size__", 50)
    run_list = ordered_chunks[: sample_size] if small_batch_mode else ordered_chunks
    total = len(run_list)

    # 控制并发：按限制包装的任务一起 gather
    results = await asyncio.gather(*[_process_single_content(c) for c in run_list])
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst


# 创建别名以保持向后兼容
extract_entities = extract_entities_dspy
