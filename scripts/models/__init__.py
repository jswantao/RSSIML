# -*- coding: utf-8 -*-
"""统一模型模块 — SVM / CNN 身份认证模型、评估与工具。

此包提供身份认证任务所需的全部模型组件:
  - SVM: 在线/离线 SVM 二分类验证器
  - CNN: 轻量 1D-CNN 二分类器
  - Metrics: 认证评估指标 (FAR/FRR/HTER/EER)
  - Memory: GPU/系统内存监控
  - Logging: 训练日志记录
"""
from scripts.models.svm import (
    SVMConfig,
    AuthenticationModel,
    SVMAuthenticationTrainer,
    svm_scores,
    compute_threshold,
)
from scripts.models.cnn.trainer import (
    CNNTrainConfig,
    CNNTrainer,
    CNNInference,
)
from scripts.models.cnn.models import (
    CNNConfig,
    RSSICNNBinaryClassifier,
    CNNAuthenticationModel,
)
from scripts.models.metrics import (
    MetricsCalculator,
    evaluate_authentication,
    TrainingError,
)
from scripts.models.memory import (
    get_memory_monitor,
    clear_gpu_memory,
)
from scripts.models.logging import (
    log_training,
)
from scripts.models.checkpoint import (
    load_cnn_checkpoint,
    train_cnn_authentication,
)

__all__ = [
    # SVM
    "SVMConfig",
    "AuthenticationModel",
    "SVMAuthenticationTrainer",
    "svm_scores",
    "compute_threshold",
    # CNN Config / Models
    "CNNConfig",
    "CNNTrainConfig",
    "RSSICNNBinaryClassifier",
    "CNNAuthenticationModel",
    # CNN Trainer / Inference
    "CNNTrainer",
    "CNNInference",
    # Metrics
    "MetricsCalculator",
    "evaluate_authentication",
    "TrainingError",
    # Memory
    "get_memory_monitor",
    "clear_gpu_memory",
    # Logging
    "log_training",
    # Checkpoint
    "load_cnn_checkpoint",
    "train_cnn_authentication",
]
