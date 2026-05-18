# -*- coding: utf-8 -*-
"""CNN 子包 — 1D-CNN 身份认证模型。

组件分工:
  - models.py:     模型结构定义 (RSSICNNBinaryClassifier, CNNConfig)
  - trainer.py:    训练器 (CNNTrainer)
  - inference.py:  推理器 (CNNInference) — 从检查点加载模型
  - utils.py:      便捷函数 (load_cnn_checkpoint, train_cnn_authentication)

CNNConfig (模型结构参数)   → cnn/models.py
CNNTrainConfig (训练超参数) → models/config.py
"""
from scripts.models.cnn.models import (
    CNNConfig,
    RSSICNNBinaryClassifier,
    CNNAuthenticationModel,
)
from scripts.models.cnn.trainer import CNNTrainer
from scripts.models.config import CNNTrainConfig

__all__ = [
    "CNNConfig",
    "RSSICNNBinaryClassifier",
    "CNNAuthenticationModel",
    "CNNTrainConfig",
    "CNNTrainer",
]
