"""
ModernBERT 分类器适配器

说明：
- 本适配器为实验一脚本提供 `ModernBertClassifier` 类，接口与脚本一致；
- 内部复用 `BertClassifier`/`TransformerClassifier` 的训练与推理流程，
  仅将默认模型名替换为 ModernBERT，并暴露一致的方法签名；
- 这样避免在实验脚本中直接依赖核心库训练器，实现阶段内自洽且可复现。
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .base_classifiers import BertClassifier


class ModernBertClassifier(BertClassifier):
    """ModernBERT 分类器适配器

    继承自 `BertClassifier`，保持训练/预测 API 一致，
    仅在构造时将默认 `model_name` 替换为 ModernBERT 对应权重名，
    并透传训练相关超参数（学习率、batch size、weight decay、warmup、epochs 等）。
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        num_classes: int = 3,
        max_length: int = 512,
        device: str = "auto",
        use_local_model: bool = True,
        fallback_to_online: bool = True,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_epochs: int = 5,
        dropout: float = 0.1,
        classifier_hidden_layers: Optional[List[int]] = None,
        early_stopping_patience: int = 2,
        use_cross_validation: bool = False,
        n_folds: int = 1,
        random_state: int = 42,
    ) -> None:
        # 记录训练相关默认值，以便 fit 时使用
        self.default_learning_rate = learning_rate
        self.default_batch_size = batch_size
        self.default_weight_decay = weight_decay
        self.default_warmup_steps = warmup_steps
        self.default_num_epochs = num_epochs
        self.default_dropout = dropout
        self.default_classifier_hidden_layers = classifier_hidden_layers
        self.early_stopping_patience = early_stopping_patience
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.random_state = random_state

        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            max_length=max_length,
            device=device,
            use_local_model=use_local_model,
            fallback_to_online=fallback_to_online,
        )

    # 为保持脚本兼容性，覆写 fit，接受与脚本一致的签名
    def fit(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
    ) -> "ModernBertClassifier":
        """训练 ModernBERT 分类器（内部复用 BertClassifier.fit）

        参数全部来自初始化时设置的默认值，保证与超参搜索解耦。
        """
        validation_tuple = (X_val, y_val) if X_val is not None and y_val is not None else None

        super().fit(
            X_train,
            y_train,
            learning_rate=self.default_learning_rate,
            batch_size=self.default_batch_size,
            max_epochs=self.default_num_epochs,
            validation_data=validation_tuple,
            early_stopping=True,
            patience=self.early_stopping_patience,
            weight_decay=self.default_weight_decay,
        )
        return self

    # 其余 API 直接继承：predict, predict_proba, get_logits

