# -*- coding: utf-8 -*-
"""CNN 便捷函数 — 轻量封装。"""
from typing import Optional

from scripts.config import PipelineConfig
from scripts.models.cnn.inference import CNNInference
from scripts.models.cnn.trainer import CNNTrainer
from scripts.models.config import CNNTrainConfig


def load_cnn_checkpoint(path):
    return CNNInference(path)


def train_cnn_authentication(data_file=None, config: Optional[PipelineConfig] = None, **kw):
    return CNNTrainer(
        train_config=CNNTrainConfig(**{k: v for k, v in kw.items() if v is not None}),
        config=config,
    ).train_authentication(data_file)
