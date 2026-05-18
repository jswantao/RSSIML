# -*- coding: utf-8 -*-
"""CNN 检查点管理 — 加载/保存/恢复。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_cnn_checkpoint(path: Path | str) -> dict[str, Any]:
    """加载 CNN 检查点文件。"""
    try:
        import torch
        checkpoint = torch.load(
            Path(path), map_location="cpu", weights_only=False,
        )
        logger.info("CNN 检查点已加载: %s", path)
        return checkpoint
    except ImportError:
        raise ImportError("PyTorch 不可用, 无法加载 CNN 检查点")
    except Exception as e:
        logger.error("CNN 检查点加载失败: %s — %s", path, e)
        raise


def train_cnn_authentication(
    data_file: Path | str,
    model_path: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """便捷函数 — 训练 CNN 认证模型。"""
    from scripts.models.cnn.trainer import CNNTrainer, CNNTrainConfig
    from scripts.models.cnn.models import CNNConfig

    train_config = CNNTrainConfig(
        epochs=kwargs.get("epochs", 20),
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("learning_rate", 0.001),
    )
    model_cfg = CNNConfig(
        use_checkpoint=kwargs.get("use_checkpoint", True),
    )

    trainer = CNNTrainer(model_cfg=model_cfg, train_config=train_config)
    return trainer.train_authentication(
        data_file=Path(data_file),
        model_path=Path(model_path) if model_path else None,
    )
