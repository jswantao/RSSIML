# -*- coding: utf-8 -*-
"""CNN 便捷函数 — 轻量封装。"""
from __future__ import annotations
from pathlib import Path

from scripts.config import PipelineConfig
from scripts.models.cnn.inference import CNNInference
from scripts.models.cnn.trainer import CNNTrainer
from scripts.models.config import CNNTrainConfig


def load_cnn_checkpoint(path: str | Path) -> CNNInference:
    return CNNInference(path)


def train_cnn_authentication(data_file=None, config: PipelineConfig | None = None, **kw):
    return CNNTrainer(
        train_config=CNNTrainConfig(**{k: v for k, v in kw.items() if v is not None}),
        config=config,
    ).train_authentication(data_file)
