import json
import dspy
from pydantic import BaseModel, Field, ValidationError
from nano_graphrag._utils import clean_str
from nano_graphrag._utils import logger
from nano_graphrag._utils import convert_response_to_json


"""
Obtained from:
https://github.com/SciPhi-AI/R2R/blob/6e958d1e451c1cb10b6fc868572659785d1091cb/r2r/providers/prompts/defaults.jsonl
"""
ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENTAGE",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "NATIONALITY",
    "RELIGION",
    "TITLE",
    "PROFESSION",
    "ANIMAL",
    "PLANT",
    "DISEASE",
    "MEDICATION",
    "CHEMICAL",
    "MATERIAL",
    "COLOR",
    "SHAPE",
    "MEASUREMENT",
    "WEATHER",
    "NATURAL_DISASTER",
    "AWARD",
    "LAW",
    "CRIME",
    "TECHNOLOGY",
    "SOFTWARE",
    "HARDWARE",
    "VEHICLE",
    "FOOD",
    "DRINK",
    "SPORT",
    "MUSIC_GENRE",
    "INSTRUMENT",
    "ARTWORK",
    "BOOK",
    "MOVIE",
    "TV_SHOW",
    "ACADEMIC_SUBJECT",
    "SCIENTIFIC_THEORY",
    "POLITICAL_PARTY",
    "CURRENCY",
    "STOCK_SYMBOL",
    "FILE_TYPE",
    "PROGRAMMING_LANGUAGE",
    "MEDICAL_PROCEDURE",
    "CELESTIAL_BODY",
]


class Entity(BaseModel):
    entity_name: str = Field(..., description="The name of the entity.")
    entity_type: str = Field(..., description="The type of the entity.")
    description: str = Field(
        ..., description="The description of the entity, in details and comprehensive."
    )
    importance_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Importance score of the entity. Should be between 0 and 1 with 1 being the most important.",
    )

    def to_dict(self):
        return {
            "entity_name": clean_str(self.entity_name.upper()),
            "entity_type": clean_str(self.entity_type.upper()),
            "description": clean_str(self.description),
            "importance_score": float(self.importance_score),
        }


class Relationship(BaseModel):
    src_id: str = Field(..., description="The name of the source entity.")
    tgt_id: str = Field(..., description="The name of the target entity.")
    description: str = Field(
        ...,
        description="The description of the relationship between the source and target entity, in details and comprehensive.",
    )
    weight: float = Field(
        ...,
        ge=0,
        le=1,
        description="The weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
    )
    order: int = Field(
        ...,
        ge=1,
        le=3,
        description="The order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order.",
    )

    def to_dict(self):
        return {
            "src_id": clean_str(self.src_id.upper()),
            "tgt_id": clean_str(self.tgt_id.upper()),
            "description": clean_str(self.description),
            "weight": float(self.weight),
            "order": int(self.order),
        }


class CombinedExtraction(dspy.Signature):
    """
    任务：给定输入文本 `input_text` 和可用实体类型列表 `entity_types`，抽取实体与实体间关系。

    输出要求（极其重要）：
    - 只输出一个 JSON 字符串（不要附加多余解释、不要包裹 Markdown 代码块）。
    - JSON 顶层必须包含两个字段：
      {"entities": [...], "relationships": [...]}。
    - entities 中的每个对象包含字段：
      - entity_name: string（从原文中抽取的原子词/短语，避免笼统词；全大写由调用方处理）
      - entity_type: string（必须出自给定的 entity_types 列表之一）
      - description: string（多句、详细）
      - importance_score: float（0-1 区间）
    - relationships 中的每个对象包含字段：
      - src_id: string（必须精确匹配某个 entities.entity_name）
      - tgt_id: string（必须精确匹配某个 entities.entity_name）
      - description: string（多句、详细）
      - weight: float（0-1 区间）
      - order: int（1、2、或 3，对应直接/二阶/三阶关系）

    质量准则（摘要）：
    - 覆盖所有重要实体并给出充分描述；
    - 关系应说明性质、影响、背景、演化、关键事件；
    - 避免引入给定列表之外的 entity_type；
    - 数值字段保持在约束范围内。
    """

    input_text: str = dspy.InputField(
        desc="The text to extract entities and relationships from."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of entity types used for extraction."
    )
    extraction_json: str = dspy.OutputField(
        desc="A pure JSON string with keys 'entities' and 'relationships'."
    )


class CritiqueCombinedExtraction(dspy.Signature):
    """
    Critique the current extraction of entities and relationships from a given text.
    Focus on completeness, accuracy, and adherence to the provided entity types and extraction guidelines.

    Critique Guidelines:
    1. Evaluate if all relevant entities from the input text are captured and correctly typed.
    2. Check if entity descriptions are comprehensive and follow the provided guidelines.
    3. Assess the completeness of relationship extractions, including higher-order relationships.
    4. Verify that relationship descriptions are detailed and follow the provided guidelines.
    5. Identify any inconsistencies, errors, or missed opportunities in the current extraction.
    6. Suggest specific improvements or additions to enhance the quality of the extraction.
    """

    input_text: str = dspy.InputField(
        desc="The original text from which entities and relationships were extracted."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of valid entity types for this extraction task."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of currently extracted entities to be critiqued."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of currently extracted relationships to be critiqued."
    )
    entity_critique: str = dspy.OutputField(
        desc="Detailed critique of the current entities, highlighting areas for improvement for completeness and accuracy.."
    )
    relationship_critique: str = dspy.OutputField(
        desc="Detailed critique of the current relationships, highlighting areas for improvement for completeness and accuracy.."
    )


class RefineCombinedExtraction(dspy.Signature):
    """
    Refine the current extraction of entities and relationships based on the provided critique.
    Improve completeness, accuracy, and adherence to the extraction guidelines.

    Refinement Guidelines:
    1. Address all points raised in the entity and relationship critiques.
    2. Add missing entities and relationships identified in the critique.
    3. Improve entity and relationship descriptions as suggested.
    4. Ensure all refinements still adhere to the original extraction guidelines.
    5. Maintain consistency between entities and relationships during refinement.
    6. Focus on enhancing the overall quality and comprehensiveness of the extraction.
    """

    input_text: str = dspy.InputField(
        desc="The original text from which entities and relationships were extracted."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of valid entity types for this extraction task."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of currently extracted entities to be refined."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of currently extracted relationships to be refined."
    )
    entity_critique: str = dspy.InputField(
        desc="Detailed critique of the current entities to guide refinement."
    )
    relationship_critique: str = dspy.InputField(
        desc="Detailed critique of the current relationships to guide refinement."
    )
    refined_entities: list[Entity] = dspy.OutputField(
        desc="List of refined entities, addressing the entity critique and improving upon the current entities."
    )
    refined_relationships: list[Relationship] = dspy.OutputField(
        desc="List of refined relationships, addressing the relationship critique and improving upon the current relationships."
    )


class TypedEntityRelationshipExtractorException(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module,
        exception_types: tuple[type[Exception]] = (Exception,),
    ):
        super().__init__()
        self.predictor = predictor
        self.exception_types = exception_types

    def copy(self):
        return TypedEntityRelationshipExtractorException(self.predictor)

    def forward(self, **kwargs):
        try:
            prediction = self.predictor(**kwargs)
            return prediction

        except Exception as e:
            if isinstance(e, self.exception_types):
                return dspy.Prediction(entities=[], relationships=[])

            raise e


class TypedEntityRelationshipExtractor(dspy.Module):
    def __init__(
        self,
        lm: dspy.LM = None,
        max_retries: int = 3,
        entity_types: list[str] = ENTITY_TYPES,
        self_refine: bool = False,
        num_refine_turns: int = 1,
    ):
        super().__init__()
        self.lm = lm
        self.entity_types = entity_types
        self.self_refine = self_refine
        self.num_refine_turns = num_refine_turns

        # 使用非结构化的字符串输出，避免触发 JSON mode/structured outputs
        self.extractor = dspy.ChainOfThought(
            signature=CombinedExtraction, max_retries=max_retries
        )
        self.extractor = TypedEntityRelationshipExtractorException(
            self.extractor, exception_types=(ValueError,)
        )

        if self.self_refine:
            self.critique = dspy.ChainOfThought(
                signature=CritiqueCombinedExtraction, max_retries=max_retries
            )
            self.refine = dspy.ChainOfThought(
                signature=RefineCombinedExtraction, max_retries=max_retries
            )

    def forward(self, input_text: str) -> dspy.Prediction:
        """
        以“纯文本 JSON 输出 + 解析”的方式进行实体/关系抽取。

        - 不依赖 DSPy 的结构化输出与 JSON mode，适配不支持 response_format 的提供方（如 DashScope 的 OpenAI 兼容端点）。
        - 使用 `convert_response_to_json` 对 LLM 返回进行鲁棒解析；
        - 使用 Pydantic 进行字段校验与归一化，再返回 `dspy.Prediction` 结构。
        """
        with dspy.context(lm=self.lm if self.lm is not None else dspy.settings.lm):
            result = self.extractor(
                input_text=input_text, entity_types=self.entity_types
            )

        raw_text = getattr(result, "extraction_json", "") or ""

        try:
            parsed = convert_response_to_json(raw_text)
        except Exception:
            # 兜底：尝试直接加载
            try:
                parsed = json.loads(raw_text)
            except Exception:
                logger.error("实体抽取解析失败：无法从模型输出中解析 JSON。")
                return dspy.Prediction(entities=[], relationships=[])

        # 取出实体与关系，做基本校验
        entities_payload = parsed.get("entities", []) if isinstance(parsed, dict) else []
        relationships_payload = (
            parsed.get("relationships", []) if isinstance(parsed, dict) else []
        )

        validated_entities: list[Entity] = []
        validated_relationships: list[Relationship] = []

        # 校验实体
        for item in entities_payload:
            if not isinstance(item, dict):
                continue
            try:
                entity = Entity(**item)
                validated_entities.append(entity)
            except ValidationError:
                # 严格保真：丢弃不合格条目
                continue

        # 允许关系引用的实体名集合（统一大写后再比较）
        upper_entity_names = {e.entity_name.upper() for e in validated_entities}

        # 校验关系
        for item in relationships_payload:
            if not isinstance(item, dict):
                continue
            try:
                rel = Relationship(**item)
                # 关系两端必须在实体集合中
                if (
                    rel.src_id.upper() in upper_entity_names
                    and rel.tgt_id.upper() in upper_entity_names
                ):
                    validated_relationships.append(rel)
            except ValidationError:
                continue

        entities = [entity.to_dict() for entity in validated_entities]
        relationships = [rel.to_dict() for rel in validated_relationships]

        return dspy.Prediction(entities=entities, relationships=relationships)
