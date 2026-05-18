# -*- coding: utf-8 -*-
"""统一模型模块 — SVM / CNN 身份认证模型、评估与工具。

此包提供身份认证任务所需的全部模型组件:
  - SVM: 在线/离线 SVM 二分类验证器
  - CNN: 轻量 1D-CNN 二分类器
  - Metrics: 认证评估指标 (FAR/FRR/HTER/EER)
  - Memory: GPU/系统内存监控
  - Logging: 训练日志记录
"""
import logging as _logging

_logger = _logging.getLogger(__name__)

# ── 版本常量 (必须在所有子模块导入之前定义，打破循环引用) ────
MODEL_VERSION = "2.0"

# ── 配置 ─────────────────────────────────────────────────────────────
from scripts.models.config import SVMConfig, CNNTrainConfig
from scripts.models.cnn.models import CNNConfig

# ── 基础工具 (指标计算、阈值、评估) ──────────────────────────────────
from scripts.models.base import MetricsCalculator, evaluate_authentication, compute_threshold
from scripts.models.exceptions import TrainingError

# ── SVM ──────────────────────────────────────────────────────────────
from scripts.models.svm import (
    AuthenticationModel,
    SVMAuthenticationTrainer,
    svm_scores,
    OnlineSVMVerifier,
    train_online_verifier,
)

# ── 内存/GPU 工具 ────────────────────────────────────────────────────
from scripts.models.memory import get_memory_monitor, clear_gpu_memory

# ── 训练日志 ─────────────────────────────────────────────────────────
from scripts.models.training_log import log_training

# ── CNN (可选: PyTorch 不可用时优雅降级) ──────────────────────────────
try:
    from scripts.models.cnn.models import RSSICNNBinaryClassifier, CNNAuthenticationModel
    from scripts.models.cnn.trainer import CNNTrainer
    from scripts.models.cnn.inference import CNNInference
    from scripts.models.cnn.utils import load_cnn_checkpoint, train_cnn_authentication
except (ImportError, Exception) as _e:
    _logger.warning("CNN 模块加载失败 (PyTorch 可能不可用): %s", _e)

    # 占位符: 允许代码中引用这些名称但在运行时抛出有意义的错误
    class _CNNUnavailable:
        def __init__(self, *a, **kw):
            raise ImportError("CNN 功能不可用: 请安装 PyTorch (pip install torch)")
        def __getattr__(self, name):
            raise ImportError("CNN 功能不可用: 请安装 PyTorch (pip install torch)")

    RSSICNNBinaryClassifier = _CNNUnavailable
    CNNAuthenticationModel = _CNNUnavailable
    CNNTrainer = _CNNUnavailable
    CNNInference = _CNNUnavailable
    load_cnn_checkpoint = _CNNUnavailable
    train_cnn_authentication = _CNNUnavailable

__all__ = [
    "MODEL_VERSION",
    # 配置
    "SVMConfig", "CNNConfig", "CNNTrainConfig",
    # SVM
    "AuthenticationModel", "SVMAuthenticationTrainer",
    "svm_scores", "compute_threshold",
    "OnlineSVMVerifier", "train_online_verifier",
    # CNN
    "RSSICNNBinaryClassifier", "CNNAuthenticationModel",
    "CNNTrainer", "CNNInference",
    "load_cnn_checkpoint", "train_cnn_authentication",
    # 评估
    "MetricsCalculator", "evaluate_authentication", "TrainingError",
    # 工具
    "get_memory_monitor", "clear_gpu_memory", "log_training",
]
