"""
复杂度分类器模块
基于ModernBERT的查询复杂度分类
"""

import os
import json
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# 检查依赖
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers导入失败: {e}")
    pass



logger = logging.getLogger(__name__)

@dataclass 
class ComplexityClassifierConfig:
    """复杂度分类器配置"""
    model_path: str = "nano_graphrag/models/modernbert_complexity_classifier"
    max_length: int = 256
    confidence_threshold: float = 0.7
    device: str = "auto"
    
    # 原始标签到复杂度等级的映射
    label_to_complexity: Dict[str, str] = None
    
    
    def __post_init__(self):
        if self.label_to_complexity is None:
            self.label_to_complexity = {
                # 直接使用训练时的标签映射
                "zero_hop": "zero_hop",
                "one_hop": "one_hop", 
                "multi_hop": "multi_hop"
            }
        
        # 校准功能已移除

class ComplexityClassifier:
    """复杂度分类器"""
    
    def __init__(self, config: ComplexityClassifierConfig = None):
        """初始化分类器"""
        self.config = config or ComplexityClassifierConfig()
        self.model = None
        self.tokenizer = None
        self.pkl_model = None
        self._model_type = "lora"  # lora, pkl, heuristic
        self.id2label = {}
        
        
        # 检查依赖
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers库未安装，将仅使用启发式规则")
            return
            
        # 自动加载模型
        self._load_model()
    
    @classmethod
    def load_default(cls, model_path: str = None) -> "ComplexityClassifier":
        """加载默认配置的分类器"""
        config = ComplexityClassifierConfig()
        if model_path:
            config.model_path = model_path
        return cls(config)
    
    def _load_model(self):
        """加载ModernBERT模型和相关组件"""
        if not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            logger.info(f"正在加载复杂度分类器: {self.config.model_path}")
            
            # 优先检查PKL格式的训练好的模型
            pkl_model_path = "experiments/experiment1_complexity_classifier/outputs/models/modernbert_best_model.pkl"
            if os.path.exists(pkl_model_path):
                logger.info(f"发现PKL格式的训练模型，优先加载: {pkl_model_path}")
                try:
                    self.pkl_model = self._load_pkl_model_safely(pkl_model_path)
                    if self.pkl_model is not None:
                        logger.info("PKL模型加载成功，具备完整预测接口")
                        logger.info(f"模型类型: {type(self.pkl_model)}")
                        self._model_type = "pkl"
                        return
                    else:
                        logger.warning("PKL模型加载失败，回退到LoRA模型")
                except Exception as e:
                    logger.warning(f"PKL模型加载失败: {e}，回退到LoRA模型")
                    import traceback
                    logger.debug(f"详细错误: {traceback.format_exc()}")
            
            # 检查LoRA模型路径是否存在
            if not os.path.exists(self.config.model_path):
                logger.warning(f"模型路径不存在: {self.config.model_path}，将仅使用启发式规则")
                return
                
            # 检查是否为PEFT/LoRA模型
            adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("检测到PEFT/LoRA模型，加载适配器...")
                
                # 加载PEFT配置
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                
                # 修复相对路径问题
                base_model_path = peft_config.base_model_name_or_path
                if base_model_path.startswith("../"):
                    # 处理相对路径，将其转换为绝对路径
                    adapter_dir = os.path.dirname(self.config.model_path)
                    base_model_path = os.path.abspath(os.path.join(adapter_dir, base_model_path))
                    logger.info(f"修复相对路径: {peft_config.base_model_name_or_path} -> {base_model_path}")

                # 检查修复后的路径是否存在
                if not os.path.exists(base_model_path):
                    logger.warning(f"基础模型路径不存在: {base_model_path}，将仅使用启发式规则")
                    return

                # 加载基础模型和tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=3,  # zero_hop, one_hop, multi_hop
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # 加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                self._model_type = "lora"
                
            else:
                # 普通模型加载
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._model_type = "lora"
            
            # 设置设备
            if self.config.device == "auto":
                self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.config.device)
            self.model.eval()
            
            # 设置标签映射
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
            else:
                self.id2label = {0: "zero_hop", 1: "one_hop", 2: "multi_hop"}
            
            logger.info("复杂度分类器加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.warning("将仅使用启发式规则进行分类")
    
    def _load_pkl_model_safely(self, pkl_model_path: str):
        """安全加载PKL模型 - 完整解决方案"""
        import sys
        import importlib
        import pickle
        
        try:
            # 设置环境变量避免Triton问题
            os.environ['PYTORCH_DISABLE_TRITON'] = '1'
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            
            # 1. 添加实验目录到Python路径
            experiment_dir = os.path.join(os.getcwd(), "experiments", "experiment1_complexity_classifier")
            if os.path.exists(experiment_dir) and experiment_dir not in sys.path:
                sys.path.insert(0, experiment_dir)
                logger.debug(f"临时添加路径: {experiment_dir}")
            
            # 2. 导入必要的模块
            try:
                import src.models.modernbert_classifier
                importlib.reload(src.models.modernbert_classifier)
                logger.debug("成功导入实验环境的ModernBertClassifier")
            except ImportError as e:
                logger.warning(f"无法导入实验环境模块: {e}")
                return None
            
            # 3. 使用pickle加载模型对象
            with open(pkl_model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 4. 修复配置问题 - 修复正确的transformer配置
            if hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'config'):
                transformer_config = model.model.transformer.config
                logger.debug("修复Transformer配置...")
                
                setattr(transformer_config, 'output_attentions', False)
                setattr(transformer_config, 'output_hidden_states', False)
                setattr(transformer_config, 'return_dict', True)
                setattr(transformer_config, 'use_cache', True)
                
                logger.debug("Transformer配置修复完成")
            
            # 5. 设置为评估模式
            if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                model.model.eval()
                logger.debug("模型设置为评估模式")
            
            # 6. 验证模型接口
            if hasattr(model, 'predict'):
                logger.info("PKL模型加载成功，具备完整预测接口")
                logger.info(f"模型类型: {type(model)}")
                return model
            else:
                logger.warning("PKL模型缺少predict方法")
                return None
                
        except Exception as e:
            logger.error(f"PKL模型加载失败: {e}")
            return None
    
    def _predict_with_pkl_model(self, query: str) -> str:
        """使用PKL模型进行预测 - 处理实际返回格式"""
        try:
            # 直接调用模型的predict方法
            prediction = self.pkl_model.predict(query)
            
            # 处理不同的返回格式
            if isinstance(prediction, (list, tuple, np.ndarray)) and len(prediction) > 0:
                # 获取第一个元素（可能是数组索引）
                class_idx = prediction[0]
                if isinstance(class_idx, (int, np.integer)):
                    # 映射到类别名称
                    class_names = ['zero_hop', 'one_hop', 'multi_hop']
                    if 0 <= class_idx < len(class_names):
                        return class_names[class_idx]
                    else:
                        logger.warning(f"无效的类别索引: {class_idx}")
                        return "one_hop"
                else:
                    return str(class_idx)
            elif isinstance(prediction, str):
                return prediction
            else:
                logger.warning(f"PKL模型返回未知格式: {prediction}")
                return "one_hop"  # 默认返回
                
        except Exception as e:
            logger.error(f"PKL模型预测失败: {e}")
            raise e
    
    def _get_pkl_model_probabilities(self, query: str) -> Dict[str, float]:
        """获取PKL模型的概率分布 - 处理实际返回格式"""
        try:
            if hasattr(self.pkl_model, 'predict_proba'):
                proba = self.pkl_model.predict_proba(query)
                
                # 处理不同的返回格式
                if isinstance(proba, (list, tuple, np.ndarray)) and len(proba) > 0:
                    if isinstance(proba[0], (list, tuple, np.ndarray)):
                        proba = proba[0]  # 取第一个样本的概率
                
                # 确保proba是可索引的
                if hasattr(proba, '__len__') and len(proba) >= 3:
                    return {
                        "zero_hop": float(proba[0]),
                        "one_hop": float(proba[1]),
                        "multi_hop": float(proba[2])
                    }
                else:
                    logger.warning(f"概率数组长度不足: {len(proba) if hasattr(proba, '__len__') else 'unknown'}")
                    return {}
            else:
                return {}
        except Exception as e:
            logger.debug(f"获取概率分布失败: {e}")
            return {}
    
    def _fix_pkl_model_config(self):
        """修复PKL模型的配置问题"""
        try:
            if hasattr(self.pkl_model, 'model') and hasattr(self.pkl_model.model, 'config'):
                config = self.pkl_model.model.config
                
                # 直接设置属性，不检查是否存在
                setattr(config, 'output_attentions', False)
                setattr(config, 'output_hidden_states', False)
                setattr(config, 'return_dict', True)
                setattr(config, 'use_cache', True)
                
                # 添加其他可能需要的属性
                setattr(config, 'pad_token_id', getattr(config, 'pad_token_id', 0))
                setattr(config, 'eos_token_id', getattr(config, 'eos_token_id', 2))
                
                logger.debug("PKL模型配置修复完成")
                
                # 如果模型有tokenizer，也修复tokenizer的配置
                if hasattr(self.pkl_model, 'tokenizer') and self.pkl_model.tokenizer:
                    tokenizer = self.pkl_model.tokenizer
                    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is None:
                        tokenizer.model_max_length = 512
                
        except Exception as e:
            logger.warning(f"配置修复失败: {e}")
    

    
    def load_model(self, model_path: str = None):
        """公共模型加载方法"""
        if model_path:
            self.config.model_path = model_path
        self._load_model()
            
        try:
            logger.info(f"正在加载复杂度分类器: {self.config.model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.config.model_path):
                logger.warning(f"模型路径不存在: {self.config.model_path}，将仅使用启发式规则")
                return
                
            # 检查是否为PEFT/LoRA模型
            adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("检测到PEFT/LoRA模型，加载适配器...")
                
                # 加载PEFT配置
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                
                # 修复相对路径问题
                base_model_path = peft_config.base_model_name_or_path
                if base_model_path.startswith("../"):
                    # 处理相对路径，将其转换为绝对路径
                    adapter_dir = os.path.dirname(self.config.model_path)
                    base_model_path = os.path.abspath(os.path.join(adapter_dir, base_model_path))
                    logger.info(f"修复相对路径: {peft_config.base_model_name_or_path} -> {base_model_path}")

                # 检查修复后的路径是否存在
                if not os.path.exists(base_model_path):
                    # 尝试使用项目内的基础模型
                    project_base_model_path = "nano_graphrag/models/modernbert/answerdotai_ModernBERT-large"
                    if os.path.exists(project_base_model_path):
                        base_model_path = project_base_model_path
                        logger.info(f"使用项目内基础模型: {base_model_path}")
                    else:
                        logger.warning(f"基础模型路径不存在: {base_model_path}，尝试使用在线模型")
                        base_model_path = "answerdotai/ModernBERT-large"

                # 加载基础模型
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=3,  # 我们的训练模型使用3个类别
                    device_map=self.config.device if self.config.device != "auto" else "auto",
                    attn_implementation="eager",  # 禁用Flash Attention
                    torch_dtype=torch.float32,   # 使用float32避免某些设备问题
                    trust_remote_code=True       # 允许远程代码执行
                )
                
                # 加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                
                # 加载标签映射
                label_mapping_path = os.path.join(self.config.model_path, "label_mapping.json")
                if os.path.exists(label_mapping_path):
                    with open(label_mapping_path, 'r', encoding='utf-8') as f:
                        label_mapping = json.load(f)
                    self.id2label = label_mapping.get("id2label", {})
                    
                    # 确保id2label的键是整数
                    self.id2label = {int(k): v for k, v in self.id2label.items()}
                else:
                    # 默认标签映射
                    self.id2label = {0: "zero_hop", 1: "one_hop", 2: "multi_hop"}
            else:
                # 普通模型加载
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_path,
                    device_map=self.config.device if self.config.device != "auto" else "auto",
                    attn_implementation="eager",  # 禁用Flash Attention
                    torch_dtype=torch.float32,   # 使用float32避免某些设备问题
                    trust_remote_code=True       # 允许远程代码执行
                )
                
                # 获取标签映射
                self.id2label = self.model.config.id2label
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            logger.info("复杂度分类器加载成功")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def _map_to_complexity(self, original_label: str) -> str:
        """将原始标签映射到复杂度等级"""
        # 首先尝试配置映射
        mapped = self.config.label_to_complexity.get(original_label, original_label)
        if mapped != original_label:
            return mapped
        
        # 处理LABEL_x格式的标签（ModernBERT模型输出）
        label_mapping = {
            "LABEL_0": "zero_hop",
            "LABEL_1": "one_hop", 
            "LABEL_2": "multi_hop"
        }
        if original_label in label_mapping:
            return label_mapping[original_label]
        
        # 处理其他格式的标签
        base_mapping = {
            "zero-hop": "zero_hop",
            "one-hop": "one_hop", 
            "multi-hop": "multi_hop",
            "0": "zero_hop",
            "1": "one_hop",
            "2": "multi_hop"
        }
        return base_mapping.get(original_label, original_label)
    
    def _smart_map_base_model(self, original_label: str) -> str:
        """智能映射基础模型的标签"""
        # 基础模型的标签映射
        base_mapping = {
            "zero-hop": "zero_hop",
            "one-hop": "one_hop", 
            "multi-hop": "multi_hop"
        }
        return base_mapping.get(original_label, original_label)
    
    def predict(self, query: str) -> str:
        """预测查询复杂度"""
        if not self.is_available():
            # 回退到启发式规则
            return self._heuristic_classify(query)
        
        try:
            # PKL模型预测
            if self._model_type == "pkl" and self.pkl_model is not None:
                predicted_label = self._predict_with_pkl_model(query)
                return self._map_to_complexity(predicted_label)
            
            # LoRA模型预测
            elif self._model_type == "lora" and self.model is not None:
                # 编码输入
                inputs = self.tokenizer(
                    query,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # 自动检测设备并移动输入
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_id = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_id].item()
                
                # 获取预测标签
                predicted_label = self.id2label.get(predicted_id, "one_hop")
                
                # 映射到复杂度等级
                complexity = self._map_to_complexity(predicted_label)
                
                return complexity
            else:
                return self._heuristic_classify(query)
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return self._heuristic_classify(query)
    
    def predict_with_confidence(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """预测查询复杂度（带置信度校准）"""
        if not self.is_available():
            # 回退到启发式规则
            complexity = self._heuristic_classify(query)
            return complexity, 0.5, {}
        
        try:
            # PKL模型预测
            if self._model_type == "pkl" and self.pkl_model is not None:
                predicted_label = self._predict_with_pkl_model(query)
                complexity = self._map_to_complexity(predicted_label)
                
                # 获取概率分布
                probabilities = self._get_pkl_model_probabilities(query)
                if probabilities:
                    raw_confidence = max(probabilities.values())
                else:
                    raw_confidence = 0.8  # 默认置信度
                    
            # LoRA模型预测
            elif self._model_type == "lora" and self.model is not None:
                # 编码输入
                inputs = self.tokenizer(
                    query,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # 自动检测设备并移动输入
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities_tensor = torch.softmax(logits, dim=-1)
                    predicted_id = torch.argmax(probabilities_tensor, dim=-1).item()
                    raw_confidence = probabilities_tensor[0][predicted_id].item()
                
                # 获取预测标签
                predicted_label = self.id2label.get(predicted_id, "one_hop")
                
                # 映射到复杂度等级
                complexity = self._map_to_complexity(predicted_label)
                
                # 构建概率字典
                prob_array = probabilities_tensor[0].cpu().numpy()
                probabilities = {
                    "zero_hop": float(prob_array[0]) if len(prob_array) > 0 else 0.0,
                    "one_hop": float(prob_array[1]) if len(prob_array) > 1 else 0.0,
                    "multi_hop": float(prob_array[2]) if len(prob_array) > 2 else 0.0
                }
            else:
                complexity = self._heuristic_classify(query)
                return complexity, 0.5, {}
            
            # 映射到复杂度等级
            complexity = self._map_to_complexity(predicted_label) if 'predicted_label' in locals() else complexity
            
            # 直接使用原始置信度（校准功能已移除）
            confidence = raw_confidence
            
            return complexity, confidence, probabilities
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            complexity = self._heuristic_classify(query)
            return complexity, 0.5, {}
    
    def get_logits(self, query: str) -> List[float]:
        """
        获取查询的原始logits输出
        
        Args:
            query: 查询字符串
            
        Returns:
            logits: 原始logits输出
        """
        if not self.is_available():
            # 如果模型不可用，返回一个默认的logits
            return [0.33, 0.34, 0.33]  # 均等概率
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 自动检测设备并移动输入
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0].tolist()
            
            return logits
            
        except Exception as e:
            logger.error(f"获取logits失败: {e}")
            return [0.33, 0.34, 0.33]  # 均等概率
    
    def _heuristic_classify(self, query: str) -> str:
        """启发式复杂度分类"""
        query_lower = query.lower()
        
        # 简单的启发式规则
        if len(query.split()) <= 3:
            return "zero_hop"
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity"]):
            return "multi_hop"
        else:
            return "one_hop"
    
    async def apredict(self, query: str) -> str:
        """异步预测查询复杂度"""
        return self.predict(query)
    
    def is_available(self) -> bool:
        """检查分类器是否可用"""
        if self._model_type == "pkl":
            return self.pkl_model is not None
        elif self._model_type == "lora":
            return self.model is not None and self.tokenizer is not None
        else:
            return False

    # 置信度校准功能已移除，保持基础分类功能

def get_global_classifier(model_path: str = None) -> ComplexityClassifier:
    """获取全局分类器实例"""
    global _global_classifier
    if not hasattr(get_global_classifier, '_global_classifier'):
        get_global_classifier._global_classifier = ComplexityClassifier.load_default(model_path)
    return get_global_classifier._global_classifier

async def classify_query_complexity(query: str, model_path: str = None) -> str:
    """异步分类查询复杂度"""
    classifier = get_global_classifier(model_path)
    return classifier.predict(query)

def classify_query_complexity_sync(query: str, model_path: str = None) -> str:
    """同步分类查询复杂度"""
    classifier = get_global_classifier(model_path)
    return classifier.predict(query) 