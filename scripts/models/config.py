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


@dataclass(frozen=True)
class CNNConfig:
    """CNN 模型结构配置。"""
    conv_channels: tuple[int, ...] = (64, 128, 256, 512)
    kernel_size: int = 5
    padding: int = 3
    hidden_units: int = 512
    dropout_rates: tuple[float, float] = (0.3, 0.5)
    window_size: int = 200
    use_batch_norm: bool = True
    activation: str = "relu"
    # 新增：模型压缩选项
    use_depthwise: bool = False  # 使用深度可分离卷积减少参数
    use_checkpoint: bool = True   # 梯度检查点 (省显存, 牺牲 ~40% 训练速度)

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数")
        if self.activation not in ("relu", "gelu", "leaky_relu"):
            raise ValueError(f"activation 无效: {self.activation}")

    def with_window_size(self, ws: int) -> "CNNConfig":
        return CNNConfig(
            self.conv_channels, self.kernel_size, self.padding,
            self.hidden_units, self.dropout_rates, ws,
            self.use_batch_norm, self.activation, self.use_depthwise,
        )


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
