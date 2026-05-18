# -*- coding: utf-8 -*-
"""1D-CNN 模型定义 — 轻量级身份认证二分类器。

架构:
  Input (N, C, W) → Conv1d blocks → AdaptiveAvgPool → FC → sigmoid
  支持梯度检查点 (gradient checkpointing) 降低显存占用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch 不可用, CNN 功能将禁用")


@dataclass
class CNNConfig:
    """CNN 模型架构配置。"""
    conv_channels: tuple[int, ...] = (64, 128, 256, 512)
    hidden_units: int = 512
    dropout: float = 0.3
    use_checkpoint: bool = True
    kernel_size: int = 5


class RSSICNNBinaryClassifier(nn.Module if _TORCH_AVAILABLE else object):
    """轻量 1D-CNN 二分类器 — 用于逐用户身份认证。

    输入: (batch, n_channels, window_size)
    输出: (batch, 1) sigmoid 概率
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        config: CNNConfig | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch 不可用, 无法创建 CNN 模型")

        super().__init__()
        self.config = config or CNNConfig()
        self.use_checkpoint = self.config.use_checkpoint

        channels = self.config.conv_channels
        ks = self.config.kernel_size
        pad = ks // 2

        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, ks, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], self.config.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_units, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.use_checkpoint and self.training:
            x = torch_checkpoint(self.conv_layers, x, use_reentrant=False)
        else:
            x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


@dataclass
class CNNAuthenticationModel:
    """CNN 认证模型容器。"""
    model_path: str = ""
    input_channels: int = 0
    config: CNNConfig | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    subjects: list[str] = field(default_factory=list)
    pca_model: Any = None
    scaler_model: Any = None
    feature_config: dict[str, Any] | None = None
    feature_dim: int | None = None
