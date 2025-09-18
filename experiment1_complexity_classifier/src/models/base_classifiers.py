"""
基线分类器实现

包含以下基线模型:
1. RandomClassifier: 随机分类器
2. RuleBasedClassifier: 基于规则的分类器  
3. BertClassifier: BERT-base分类器
4. RobertaClassifier: RoBERTa-large分类器
"""

import os
import random
import re
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel
)
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

# 设置logger
logger = logging.getLogger(__name__)


class RandomClassifier(BaseEstimator, ClassifierMixin):
    """随机分类器基线
    
    随机预测查询复杂度，用作最简单的基线对比。
    """
    
    def __init__(self, num_classes: int = 3, random_state: int = 42):
        """
        Args:
            num_classes: 类别数量 (zero_hop=0, one_hop=1, multi_hop=2)
            random_state: 随机种子
        """
        self.num_classes = num_classes
        self.random_state = random_state
        self.classes_ = np.arange(num_classes)
        self.class_names = ['zero_hop', 'one_hop', 'multi_hop']
        
    def fit(self, X, y=None):
        """训练模型（随机分类器无需训练）"""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        return self
    
    def predict(self, X) -> np.ndarray:
        """预测类别"""
        n_samples = len(X)
        return np.random.choice(self.classes_, size=n_samples)
    
    def predict_proba(self, X) -> np.ndarray:
        """预测概率（均匀分布）"""
        n_samples = len(X)
        # 生成随机概率并归一化
        probs = np.random.random((n_samples, self.num_classes))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def get_logits(self, X) -> np.ndarray:
        """获取logits（从概率反推）"""
        probs = self.predict_proba(X)
        # 添加小的噪声避免log(0)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return np.log(probs)


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """基于规则的分类器
    
    使用关键词、查询长度和语法模式来判断复杂度。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 规则配置字典
        """
        self.config = config or self._default_config()
        self.classes_ = np.array([0, 1, 2])  # zero_hop, one_hop, multi_hop
        self.class_names = ['zero_hop', 'one_hop', 'multi_hop']
        
    def _default_config(self) -> Dict:
        """默认规则配置"""
        return {
            'keywords': {
                'zero_hop': [
                    'what is', 'define', 'meaning', 'definition',
                    'who is', 'when was', 'where is'
                ],
                'one_hop': [
                    'who', 'when', 'where', 'which', 'how many',
                    'list', 'name', 'identify'
                ],
                'multi_hop': [
                    'compare', 'relationship', 'analyze', 'both',
                    'and', 'or', 'between', 'difference', 'similar'
                ]
            },
            'length_thresholds': {
                'short': 5,   # <= 5 words: likely zero-hop
                'medium': 15, # 6-15 words: likely one-hop  
                'long': 25    # > 15 words: likely multi-hop
            },
            'patterns': {
                'question_words': r'\b(what|who|when|where|why|how|which)\b',
                'comparison': r'\b(compare|versus|vs|difference|similar)\b',
                'multiple_entities': r'\band\b|\bor\b',
                'superlatives': r'\b(most|least|best|worst|largest|smallest)\b'
            }
        }
    
    def fit(self, X, y=None):
        """训练模型（规则分类器无需训练）"""
        return self
    
    def _extract_features(self, query: str) -> Dict:
        """从查询中提取特征"""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        features = {
            'length': len(words),
            'has_question_word': bool(re.search(self.config['patterns']['question_words'], query_lower)),
            'has_comparison': bool(re.search(self.config['patterns']['comparison'], query_lower)),
            'has_multiple_entities': bool(re.search(self.config['patterns']['multiple_entities'], query_lower)),
            'has_superlatives': bool(re.search(self.config['patterns']['superlatives'], query_lower)),
            'keyword_matches': {
                'zero_hop': sum(1 for kw in self.config['keywords']['zero_hop'] if kw in query_lower),
                'one_hop': sum(1 for kw in self.config['keywords']['one_hop'] if kw in query_lower),
                'multi_hop': sum(1 for kw in self.config['keywords']['multi_hop'] if kw in query_lower)
            }
        }
        
        return features
    
    def _rule_based_predict(self, query: str) -> int:
        """基于规则预测单个查询的复杂度"""
        features = self._extract_features(query)
        
        # 规则1: 关键词匹配
        keyword_scores = features['keyword_matches']
        max_keyword_category = max(keyword_scores.items(), key=lambda x: x[1])
        
        # 规则2: 长度判断
        length = features['length']
        if length <= self.config['length_thresholds']['short']:
            length_category = 'zero_hop'
        elif length <= self.config['length_thresholds']['medium']:
            length_category = 'one_hop'
        else:
            length_category = 'multi_hop'
        
        # 规则3: 语法模式
        if features['has_comparison'] or features['has_multiple_entities']:
            pattern_category = 'multi_hop'
        elif features['has_question_word']:
            pattern_category = 'one_hop'
        else:
            pattern_category = 'zero_hop'
        
        # 综合决策（优先级：关键词 > 语法模式 > 长度）
        if max_keyword_category[1] > 0:  # 有关键词匹配
            predicted_category = max_keyword_category[0]
        elif features['has_comparison'] or features['has_multiple_entities']:
            predicted_category = 'multi_hop'
        else:
            predicted_category = length_category
        
        # 映射到数值标签
        category_mapping = {'zero_hop': 0, 'one_hop': 1, 'multi_hop': 2}
        return category_mapping[predicted_category]
    
    def predict(self, X) -> np.ndarray:
        """预测类别"""
        predictions = []
        for query in X:
            pred = self._rule_based_predict(query)
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_proba(self, X) -> np.ndarray:
        """预测概率（基于置信度启发式）"""
        predictions = self.predict(X)
        n_samples = len(X)
        probs = np.zeros((n_samples, 3))
        
        for i, pred in enumerate(predictions):
            features = self._extract_features(X[i])
            
            # 基于特征匹配度计算置信度
            keyword_matches = features['keyword_matches']
            max_matches = max(keyword_matches.values())
            
            if max_matches > 0:
                confidence = min(0.9, 0.6 + max_matches * 0.1)
            else:
                confidence = 0.4  # 低置信度
            
            # 将置信度分配给预测类别，其余平均分配
            probs[i, pred] = confidence
            remaining_prob = (1 - confidence) / 2
            for j in range(3):
                if j != pred:
                    probs[i, j] = remaining_prob
                    
        return probs
    
    def get_logits(self, X) -> np.ndarray:
        """获取logits"""
        probs = self.predict_proba(X)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return np.log(probs)


class LocalModelLoader:
    """本地模型加载器"""
    
    # 记录已解析成功的在线候选，避免在同一进程中重复尝试失败的候选
    _resolved_online_model_cache: Dict[str, str] = {}

    @staticmethod
    def get_local_model_path(model_name: str, experiment_root: Optional[Path] = None) -> Optional[Path]:
        """获取本地模型路径"""
        if experiment_root is None:
            # 自动检测实验根目录
            current_file = Path(__file__)
            experiment_root = current_file.parent.parent.parent
        
        # 尝试从配置文件读取本地模型配置
        config_path = experiment_root / 'config' / 'model_config.yaml'
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                local_models = config.get('local_models', {})
                if local_models.get('enabled', False):
                    models_config = local_models.get('models', {})
                    if model_name in models_config:
                        local_path = experiment_root / models_config[model_name]['local_path']
                        if local_path.exists():
                            return local_path
            except Exception as e:
                logger.warning(f"读取本地模型配置失败: {e}")
        
        # 备用方法：检查标准本地模型目录（包含 ModernBERT 的规范化目录与常见错名容错）
        models_dir = experiment_root / 'models'
        model_mapping = {
            'bert-base-uncased': 'bert_base_uncased',
            'bert-base-cased': 'bert_base_cased',
            'roberta-large': 'roberta_large',
            # ModernBERT 规范化命名（优先）
            'answerdotai/ModernBERT-large': str(Path('modernbert') / 'answerdotai_ModernBERT-large'),
            'ModernBERT-large': str(Path('modernbert') / 'answerdotai_ModernBERT-large'),
        }
        
        # 规范化：对 ModernBERT 进行特别处理，兼容常见拼写误差目录
        def _validate_local_path(path: Path) -> Optional[Path]:
            if not path.exists():
                return None
            required_files = [
                'pytorch_model.bin', 'model.safetensors', 'tf_model.h5',
                'model.ckpt.index', 'flax_model.msgpack'
            ]
            return path if any((path / f).exists() for f in required_files) else None

        # 先用直接映射
        if model_name in model_mapping:
            candidate = _validate_local_path(models_dir / model_mapping[model_name])
            if candidate is not None:
                return candidate

        # ModernBERT 容错路径候选（处理日志里出现的误名），并在必要时迁移到规范路径
        if 'modernbert' in model_name.lower() or 'answerdotai/modernbert' in model_name.lower():
            canonical = models_dir / 'modernbert' / 'answerdotai_ModernBERT-large'
            typo_candidates = [
                canonical,
                models_dir / 'modernswerdotai_ModernBERT-large',
                models_dir / 'modernbewerdotai_ModernBERT-large',
            ]
            for p in typo_candidates:
                candidate = _validate_local_path(p)
                if candidate is None:
                    continue
                # 命中错名目录，且规范目录不存在时，尝试迁移
                if candidate != canonical and _validate_local_path(canonical) is None:
                    try:
                        import shutil
                        canonical.parent.mkdir(parents=True, exist_ok=True)
                        logger.info("检测到 ModernBERT 错名目录，正在迁移 -> %s", canonical)
                        shutil.move(str(candidate), str(canonical))
                        logger.info("迁移完成: %s -> %s", candidate, canonical)
                        return canonical
                    except Exception as move_err:
                        logger.warning("迁移 ModernBERT 目录失败，使用原路径: %s，错误: %s", candidate, move_err)
                        return candidate
                # 已是规范目录或规范目录可用
                return candidate
        
        return None
    
    @staticmethod
    def get_huggingface_model_name(model_name: str) -> str:
        """获取正确的 Hugging Face 模型名称"""
        # 为兼容不同组织命名，优先返回更通用的 ID
        # 注意：保持返回旧接口以兼容旧调用处
        if model_name == 'bert-base-uncased':
            return 'bert-base-uncased'
        if model_name == 'bert-base-cased':
            return 'bert-base-cased'
        if model_name == 'roberta-large':
            return 'roberta-large'
        return model_name

    @staticmethod
    def get_huggingface_model_candidates(model_name: str) -> list:
        """返回可尝试的多个 Hugging Face 模型名称候选，按优先级排列"""
        # 扩展候选列表，包含更多变体和回退选项
        candidates_map = {
            'bert-base-uncased': [
                'bert-base-uncased',
                'google-bert/bert-base-uncased',
                'distilbert-base-uncased',
                'bert-base-cased'
            ],
            'bert-base-cased': [
                'bert-base-cased',  # 标准名称（优先）
                'bert-base-uncased'  # 备选uncased版本
            ],
            'roberta-large': [
                'roberta-large',  # 标准名称（优先）
                'FacebookAI/roberta-large',  # 组织前缀版本
                'roberta-base',  # 轻量版回退
                'microsoft/DialoGPT-medium'  # 兼容性备选
            ]
        }
        
        # 基础处理
        base_name = LocalModelLoader.get_huggingface_model_name(model_name)
        
        # 获取候选列表，如果不在映射中，使用默认候选
        if model_name in candidates_map:
            return candidates_map[model_name]
        elif base_name in candidates_map:
            return candidates_map[base_name]
        else:
            # 默认候选：原名称 + 基础名称
            return [model_name, base_name] if model_name != base_name else [model_name]
    
    @staticmethod
    def load_model_from_path_or_name(
        model_name_or_path: str,
        config_class=AutoConfig,
        model_class=AutoModel,
        tokenizer_class=AutoTokenizer,
        fallback_to_online: bool = True
    ):
        """从本地路径或在线加载模型"""
        # 首先尝试本地路径
        local_path = LocalModelLoader.get_local_model_path(model_name_or_path)
        
        if local_path is not None:
            logger.info(f"🏠 使用本地模型: {local_path}")
            try:
                # 对于包含自定义代码的模型（如 ModernBERT），需要开启 trust_remote_code
                config = config_class.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True
                )
                model = model_class.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True
                )
                # 统一使用 AutoTokenizer，确保优先加载 Fast 版并支持自定义实现
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True, use_fast=True
                )
                return config, model, tokenizer
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")
                if not fallback_to_online:
                    raise
        
        # 回退到在线下载
        if fallback_to_online:
            # 依次尝试候选模型名称
            cache_dir = os.environ.get("HF_HOME")
            offline_mode = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            last_error: Optional[Exception] = None
            candidates = LocalModelLoader.get_huggingface_model_candidates(
                model_name_or_path
            )

            # 优先使用已解析的在线候选，避免每个fold反复尝试失败选项
            if model_name_or_path in LocalModelLoader._resolved_online_model_cache:
                resolved = LocalModelLoader._resolved_online_model_cache[model_name_or_path]
                candidates = [resolved] + [c for c in candidates if c != resolved]
            
            logger.info(f"尝试在线加载模型 {model_name_or_path}，候选列表: {candidates}")
            
            for idx, hf_model_name in enumerate(candidates):
                try:
                    logger.info(f"☁️ 尝试在线模型 [{idx+1}/{len(candidates)}]: {hf_model_name}")
                    
                    # 增加超时和重试设置，处理网络问题
                    config = config_class.from_pretrained(
                        hf_model_name,
                        cache_dir=cache_dir,
                        resume_download=True,
                        local_files_only=offline_mode,
                        trust_remote_code=True,
                        proxies=None,  # 确保不使用代理
                        use_auth_token=False,  # 不使用认证
                        force_download=False  # 允许使用缓存
                    )
                    model = model_class.from_pretrained(
                        hf_model_name,
                        config=config,
                        cache_dir=cache_dir,
                        resume_download=True,
                        local_files_only=offline_mode,
                        trust_remote_code=True,
                        proxies=None,
                        use_auth_token=False,
                        force_download=False
                    )
                    # 统一使用 AutoTokenizer，优先 Fast 版并支持自定义实现
                    tokenizer = AutoTokenizer.from_pretrained(
                        hf_model_name,
                        cache_dir=cache_dir,
                        resume_download=True,
                        local_files_only=offline_mode,
                        trust_remote_code=True,
                        proxies=None,
                        use_auth_token=False,
                        force_download=False,
                        use_fast=True,
                    )
                    logger.info(f"✅ 成功加载在线模型: {hf_model_name}")
                    # 记录成功解析的候选，后续同名请求直接使用
                    LocalModelLoader._resolved_online_model_cache[model_name_or_path] = hf_model_name
                    return config, model, tokenizer
                    
                except Exception as e:  # 尝试下一个候选
                    last_error = e
                    logger.warning(f"❌ 在线加载失败 [{idx+1}/{len(candidates)}] {hf_model_name}: {type(e).__name__}: {str(e)[:200]}")
                    
                    # 对于网络错误，快速尝试下一个候选
                    if any(keyword in str(e).lower() for keyword in ['network', 'connection', 'resolve', 'timeout', 'httpsconnection']):
                        logger.info(f"检测到网络问题，跳到下一个候选模型...")
                        continue
                    else:
                        # 非网络错误，记录详细信息
                        logger.error(f"模型特定错误 {hf_model_name}: {e}")
                        continue
            
            # 全部失败 - 提供更详细的错误信息和解决建议
            error_msg = f"""无法加载模型 '{model_name_or_path}'，已尝试所有候选: {candidates}
            
可能的解决方案：
1. 检查网络连接是否正常
2. 使用 --use_online_models 参数直接在线模式
3. 预先下载模型到本地: python scripts/download_models.py
4. 设置环境变量 HF_ENDPOINT=https://hf-mirror.com （使用镜像）

最后错误: {last_error}"""
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"无法加载模型 {model_name_or_path}：本地模型不存在且禁用在线回退")

    @staticmethod
    def load_tokenizer_only(
        model_name_or_path: str,
        tokenizer_class=AutoTokenizer,
        fallback_to_online: bool = True
    ):
        """仅加载分词器（本地优先 + 在线候选 + 解析缓存）

        说明：
        - 一些训练流程仅需要分词器，无需提前加载完整模型；
        - 复用候选解析与缓存逻辑，避免在每个fold中重复尝试失败候选；
        """
        # 先查找本地路径
        local_path = LocalModelLoader.get_local_model_path(model_name_or_path)
        if local_path is not None:
            logger.info(f"🏠 使用本地分词器: {local_path}")
            try:
                # 统一使用 AutoTokenizer，优先 Fast 版并支持自定义实现
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True, use_fast=True
                )
                return tokenizer
            except Exception as e:
                logger.warning(f"本地分词器加载失败: {e}")
                if not fallback_to_online:
                    raise
        
        if not fallback_to_online:
            raise RuntimeError(
                f"无法加载分词器 {model_name_or_path}：本地不存在且禁用在线回退"
            )

        # 在线候选 + 解析缓存
        cache_dir = os.environ.get("HF_HOME")
        offline_mode = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        last_error: Optional[Exception] = None
        candidates = LocalModelLoader.get_huggingface_model_candidates(
            model_name_or_path
        )
        if model_name_or_path in LocalModelLoader._resolved_online_model_cache:
            resolved = LocalModelLoader._resolved_online_model_cache[model_name_or_path]
            candidates = [resolved] + [c for c in candidates if c != resolved]

        logger.info(
            f"尝试在线加载分词器 {model_name_or_path}，候选列表: {candidates}"
        )
        for idx, hf_model_name in enumerate(candidates):
            try:
                logger.info(
                    f"☁️ 尝试在线分词器 [{idx+1}/{len(candidates)}]: {hf_model_name}"
                )
                # 统一使用 AutoTokenizer，优先 Fast 版并支持自定义实现
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    cache_dir=cache_dir,
                    resume_download=True,
                    local_files_only=offline_mode,
                    trust_remote_code=True,
                    proxies=None,
                    use_auth_token=False,
                    force_download=False,
                    use_fast=True,
                )
                logger.info(f"✅ 成功加载在线分词器: {hf_model_name}")
                LocalModelLoader._resolved_online_model_cache[model_name_or_path] = (
                    hf_model_name
                )
                return tokenizer
            except Exception as e:
                last_error = e
                logger.warning(
                    f"❌ 在线分词器失败 [{idx+1}/{len(candidates)}] {hf_model_name}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                continue
        raise RuntimeError(
            f"无法加载分词器 '{model_name_or_path}'，已尝试所有候选: {candidates}\n最后错误: {last_error}"
        )


class TransformerClassifier(nn.Module):
    """Transformer基础分类器类"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        classifier_hidden_layers: Optional[List[int]] = None,
        use_local_model: bool = True,
        fallback_to_online: bool = True
    ):
        """
        Args:
            model_name: 预训练模型名称
            num_classes: 分类类别数
            hidden_size: 隐藏层大小
            dropout: Dropout率
            classifier_hidden_layers: 分类头隐藏层大小列表
            use_local_model: 是否优先使用本地模型
            fallback_to_online: 如果本地模型不存在，是否回退到在线下载
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_local_model = use_local_model
        
        # 加载预训练模型（支持本地优先策略）
        if use_local_model:
            self.config, self.transformer, self.tokenizer = LocalModelLoader.load_model_from_path_or_name(
                model_name,
                fallback_to_online=fallback_to_online
            )
        else:
            # 传统在线加载方式
            cache_dir = os.environ.get("HF_HOME")
            offline_mode = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            
            self.config = AutoConfig.from_pretrained(
                model_name, 
                cache_dir=cache_dir, 
                resume_download=True, 
                local_files_only=offline_mode
            )
            self.transformer = AutoModel.from_pretrained(
                model_name, 
                config=self.config, 
                cache_dir=cache_dir, 
                resume_download=True, 
                local_files_only=offline_mode
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir, 
                resume_download=True, 
                local_files_only=offline_mode,
                trust_remote_code=True,
                use_fast=True,
            )
        
        # 推断隐藏维度：优先使用显式传入，否则从配置自动获取
        effective_hidden_size = (
            self.hidden_size
            if self.hidden_size is not None
            else getattr(self.config, "hidden_size", None)
        )
        if effective_hidden_size is None:
            effective_hidden_size = 768
        self.hidden_size = effective_hidden_size

        # 分类头
        if classifier_hidden_layers:
            layers = []
            input_size = effective_hidden_size
            for hidden_dim in classifier_hidden_layers:
                layers.extend([
                    nn.Linear(input_size, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                input_size = hidden_dim
            layers.append(nn.Linear(input_size, num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(effective_hidden_size, num_classes)
            )
        
        # 初始化分类器权重
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """初始化分类器权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        # 兼容不同架构：部分模型（如 ModernBERT）不接受 token_type_ids 参数
        # 为最大兼容性，这里不向底层模型传递 token_type_ids。
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的表示
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }


class BertClassifier(BaseEstimator, ClassifierMixin):
    """BERT-base分类器包装类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        num_classes: int = 3,
        max_length: int = 512,
        device: str = "auto",
        use_local_model: bool = True,
        fallback_to_online: bool = True
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_local_model = use_local_model
        self.fallback_to_online = fallback_to_online
        
        self.classes_ = np.array([0, 1, 2])
        self.class_names = ['zero_hop', 'one_hop', 'multi_hop']
        
        # 模型将在fit时初始化
        self.model = None
        self.tokenizer = None
        self.is_fitted = False
    
    def fit(self, X, y, learning_rate=2e-5, batch_size=16, max_epochs=5,
            validation_data=None, early_stopping=False, patience=2,
            weight_decay: float = 0.0, **kwargs):
        """训练模型 - 完整的训练流程"""
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from sklearn.metrics import accuracy_score
        import torch.nn.functional as F
        
        logger.info(f"开始训练 {self.model_name}")
        
        # 使用本地优先策略仅加载分词器（避免重复加载完整模型）
        try:
            self.tokenizer = LocalModelLoader.load_tokenizer_only(
                self.model_name,
                fallback_to_online=self.fallback_to_online
            )
        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            raise
        
        # 初始化模型
        self.model = TransformerClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            hidden_size=None,
            use_local_model=self.use_local_model,
            fallback_to_online=self.fallback_to_online
        )
        self.model.to(self.device)
        
        # 准备训练数据
        train_encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        train_labels = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            train_labels.to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 准备验证数据（如果提供）
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_encodings = self.tokenizer(
                X_val, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            )
            val_labels = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(
                val_encodings['input_ids'].to(self.device),
                val_encodings['attention_mask'].to(self.device),
                val_labels.to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 设置优化器（使 weight_decay 生效）
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger.info(
            "BertClassifier 优化器: AdamW | lr=%.2e, weight_decay=%s, batch_size=%s",
            learning_rate, weight_decay, batch_size
        )
        
        # 训练循环
        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs_completed': 0
        }
        
        # 梯度累积步数（减少显存使用）
        accumulation_steps = 2 if batch_size >= 16 else 1
        
        for epoch in range(max_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # 添加进度条
            logger.info(f'Epoch {epoch+1}/{max_epochs} - 开始训练...')
            train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False)
            
            # 梯度累积计数器
            step_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs['logits'], labels)
                
                # 归一化损失（梯度累积）
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
                
                # 计算准确率
                with torch.no_grad():
                    _, predicted = torch.max(outputs['logits'], 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                
                # 梯度累积和优化器步骤
                step_count += 1
                if step_count % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 定期清理GPU缓存
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
            
            # 处理最后剩余的梯度
            if step_count % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_acc = correct_predictions / total_predictions
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch {epoch+1}/{max_epochs} 完成 - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}')
            
            # 记录训练历史
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            
            # 验证
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = batch
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        _, predicted = torch.max(outputs['logits'], 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                
                val_acc = val_correct / val_total
                self.model.train()
                
                # 记录验证历史
                history['val_acc'].append(val_acc)
                
                # 早停机制
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if early_stopping and patience_counter >= patience:  # 早停
                        logger.info(f"早停于epoch {epoch+1}")
                        break
            
            logger.info(f"Epoch {epoch+1}/{max_epochs}: Loss={avg_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # 每个 epoch 结束清理一次显存，缓解碎片化
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        
        # 更新完成的epoch数
        history['epochs_completed'] = epoch + 1
        
        self.is_fitted = True
        
        # 返回训练历史
        return history
    
    def fit_with_params(self, X, y, params):
        """使用参数字典训练模型"""
        training_params = {
            'learning_rate': params.get('learning_rate', 2e-5),
            'batch_size': params.get('batch_size', 16),
            'max_epochs': params.get('max_epochs', 5),
            'validation_data': params.get('validation_data'),
            'early_stopping': params.get('early_stopping', False),
            'patience': params.get('patience', 2),
            'weight_decay': params.get('weight_decay', 0.0),
        }
        return self.fit(X, y, **training_params)
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """批量分词"""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def predict(self, X) -> np.ndarray:
        """预测类别"""
        logits = self.get_logits(X)
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, X) -> np.ndarray:
        """预测概率"""
        logits = self.get_logits(X)
        return torch.softmax(torch.tensor(logits), dim=1).numpy()
    
    def get_logits(self, X) -> np.ndarray:
        """获取logits"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        self.model.eval()
        all_logits = []
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i + batch_size]
                inputs = self._tokenize_batch(batch_texts)
                
                outputs = self.model(**inputs)
                logits = outputs['logits'].cpu().numpy()
                all_logits.append(logits)
        
        return np.concatenate(all_logits, axis=0)


class RobertaClassifier(BertClassifier):
    """RoBERTa-large分类器包装类"""
    
    def __init__(
        self,
        model_name: str = "roberta-large",
        num_classes: int = 3,
        max_length: int = 512,
        device: str = "auto",
        use_local_model: bool = True,
        fallback_to_online: bool = True
    ):
        super().__init__(model_name, num_classes, max_length, device, use_local_model, fallback_to_online)
    
    def fit(self, X, y, learning_rate=1e-5, batch_size=16, max_epochs=5,
            validation_data=None, early_stopping=False, patience=2,
            weight_decay: float = 0.0, **kwargs):
        """训练模型 - 完整的训练流程（带OOM自动恢复）"""
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from sklearn.metrics import accuracy_score
        import torch.nn.functional as F
        
        logger.info(f"开始训练 {self.model_name} (初始batch_size={batch_size})")
        
        # OOM恢复机制：递归重试，每次减小batch_size
        def _try_training_with_batch_size(current_batch_size, retry_count=0):
            if current_batch_size < 2:
                raise RuntimeError("批次大小已减少到最小值，仍然OOM，请释放更多显存或使用更小的模型")
            if retry_count > 3:
                raise RuntimeError("OOM重试次数过多，训练失败")
            
            try:
                return self._actual_fit(X, y, learning_rate, current_batch_size, max_epochs,
                                      validation_data, early_stopping, patience, weight_decay)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"OOM错误，batch_size从{current_batch_size}减少到{current_batch_size//2}")
                    # 清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    # 递归重试，批次大小减半
                    return _try_training_with_batch_size(current_batch_size // 2, retry_count + 1)
                else:
                    # 非OOM错误，直接抛出
                    raise
        
        return _try_training_with_batch_size(batch_size)
    
    def _actual_fit(self, X, y, learning_rate, batch_size, max_epochs,
                   validation_data, early_stopping, patience, weight_decay: float = 0.0):
        """实际的训练逻辑（从原fit方法提取）"""
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        import torch.nn.functional as F
        
        logger.info(f"实际训练使用batch_size={batch_size}")
        
        # 使用本地优先策略仅加载分词器（避免重复加载完整模型）
        try:
            self.tokenizer = LocalModelLoader.load_tokenizer_only(
                self.model_name,
                fallback_to_online=self.fallback_to_online
            )
        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            raise
        
        # 初始化模型
        self.model = TransformerClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            hidden_size=None,
            use_local_model=self.use_local_model,
            fallback_to_online=self.fallback_to_online
        )
        self.model.to(self.device)
        
        # 准备训练数据
        train_encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        train_labels = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            train_labels.to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 准备验证数据（如果提供）
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_encodings = self.tokenizer(
                X_val, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            )
            val_labels = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(
                val_encodings['input_ids'].to(self.device),
                val_encodings['attention_mask'].to(self.device),
                val_labels.to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 设置优化器（使 weight_decay 生效）
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger.info(
            "RobertaClassifier 优化器: AdamW | lr=%.2e, weight_decay=%s, batch_size=%s",
            learning_rate, weight_decay, batch_size
        )
        
        # 训练循环
        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0
        
        # 训练历史记录（与 BertClassifier 对齐，供绘图使用）
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs_completed': 0
        }
        
        # 梯度累积步数（减少显存使用）
        accumulation_steps = 2 if batch_size >= 16 else 1
        
        for epoch in range(max_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # 添加进度条
            logger.info(f'Epoch {epoch+1}/{max_epochs} - 开始训练...')
            train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False)
            
            # 梯度累积计数器
            step_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs['logits'], labels)
                
                # 归一化损失（梯度累积）
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
                
                # 计算准确率
                with torch.no_grad():
                    _, predicted = torch.max(outputs['logits'], 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                
                # 梯度累积和优化器步骤
                step_count += 1
                if step_count % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 定期清理GPU缓存
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
            
            # 处理最后剩余的梯度
            if step_count % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_acc = correct_predictions / total_predictions
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch {epoch+1}/{max_epochs} 完成 - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}')
            
            # 记录训练历史
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            
            # 验证
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = batch
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        _, predicted = torch.max(outputs['logits'], 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                
                val_acc = val_correct / val_total
                self.model.train()
                
                # 记录验证历史
                history['val_acc'].append(val_acc)
                
                # 早停机制
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if early_stopping and patience_counter >= patience:  # 早停
                        logger.info(f"早停于epoch {epoch+1}")
                        break
            
            logger.info(f"Epoch {epoch+1}/{max_epochs}: Loss={avg_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # 每个 epoch 结束清理一次显存，缓解碎片化
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.is_fitted = True
        history['epochs_completed'] = epoch + 1
        return history
    
    def fit_with_params(self, X, y, params):
        """使用参数字典训练模型"""
        training_params = {
            'learning_rate': params.get('learning_rate', 1e-5),  # RoBERTa用更小的学习率
            'batch_size': params.get('batch_size', 16),
            'max_epochs': params.get('max_epochs', 5),
            'validation_data': params.get('validation_data')
        }
        return self.fit(X, y, **training_params)