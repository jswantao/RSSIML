# -*- coding: utf-8 -*-
"""模型训练配置 — SVM 与 CNN 共享的数据类。"""
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SVMConfig:
    """SVM 训练配置。"""
    cv_folds: int = 3
    threshold_method: str = "youden"
    distance_threshold_quantile: float = 0.5
    random_seed: int = 42
    search_method: str = "grid"
    random_search_iters: int = 20
    svm_C: float = 1.0
    svm_gamma: str = "scale"  # "scale" 自适应维度, 避免小 gamma 在高维 CSI 上欠拟合
    max_memory_mb: int = 2048

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds 必须 >= 2")
        valid_threshold = {"youden", "quantile", "fixed", "eer"}
        if self.threshold_method not in valid_threshold:
            raise ValueError(
                f"threshold_method 无效: {self.threshold_method}")
        if self.search_method not in ("grid", "random"):
            raise ValueError(
                f"search_method 必须是 'grid' 或 'random'")


# CNNConfig 定义在 scripts.models.cnn.models 中，避免重复

@dataclass(frozen=True)
class CNNTrainConfig:
    """CNN 训练超参数。"""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    early_stop_patience: int = 5
    threshold_method: str = "youden"
    distance_threshold_quantile: float = 0.5
    random_seed: int = 42
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 3
    gradient_accumulation_steps: int = 4
    use_memory_efficient_attention: bool = False
    max_memory_mb: int = 4096

    def __post_init__(self) -> None:
        if not 0 < self.val_ratio < 0.5:
            raise ValueError(f"val_ratio 必须在 (0, 0.5) 之间")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs 必须 >= 0")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 必须 >= 1")


# ══════════════════════════════════════════════════════════════════════════════
