# -*- coding: utf-8 -*-
"""CNN 推理器 — 模型加载、认证预测。"""
import gc
import logging
from pathlib import Path
import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder

from scripts.models.cnn.models import (
    _ConvBackbone,
    _get_activation,
    prepare_cnn_input,
    RSSICNNBinaryClassifier,
)
from scripts.models.cnn.models import CNNConfig
from scripts.models import MODEL_VERSION

logger = logging.getLogger(__name__)


class CNNInference:
    """CNN 推理器 — 仅支持认证模型。"""

    def __init__(self, ckpt_path: Path) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # checkpoint 包含 LabelEncoder/dict 等非 tensor 对象,
        # 必须 weights_only=False; 安全前提: 仅加载自己训练产出的模型文件
        ckpt = torch.load(
            ckpt_path, map_location=self.device, weights_only=False)

        if "verifier_states" not in ckpt:
            raise ValueError("不支持非认证模型检查点。此版本仅支持身份认证。")

        self._init_auth(ckpt)
        del ckpt
        gc.collect()

    def _init_auth(self, ckpt):
        ckpt_version = ckpt.get("model_version", "0.0")
        if ckpt_version != MODEL_VERSION:
            raise ValueError(
                f"Checkpoint 版本 ({ckpt_version}) 与当前版本 ({MODEL_VERSION}) 不兼容，"
                f"请重新训练模型。")
        self.is_authentication = True
        self.task = "authentication"
        self.encoder = ckpt["label_encoder"]
        self.subjects = ckpt["subjects"]
        self.thresholds = ckpt["thresholds"]

        cfg = ckpt.get("model_config")
        if isinstance(cfg, dict):
            cfg = CNNConfig(
                conv_channels=tuple(cfg.get("conv_channels", (64, 128, 256, 512))),
                kernel_size=int(cfg.get("kernel_size", 5)),
                padding=int(cfg.get("padding", 3)),
                hidden_units=int(cfg.get("hidden_units", 512)),
                dropout_rates=tuple(cfg.get("dropout_rates", (0.3, 0.5))),
                window_size=int(cfg.get("window_size", 200)),
                use_batch_norm=bool(cfg.get("use_batch_norm", True)),
                activation=str(cfg.get("activation", "relu")),
                use_depthwise=bool(cfg.get("use_depthwise", False)),
                use_checkpoint=bool(cfg.get("use_checkpoint", True)),
            )

        in_ch = ckpt["input_channels"]
        # 修复旧版 checkpoint: input_channels 可能误存为瓶颈输出通道数 (64)
        first_sd = next(iter(ckpt["verifier_states"].values()))
        if "backbone.bottleneck.0.weight" in first_sd:
            true_in_ch = first_sd["backbone.bottleneck.0.weight"].shape[1]
            if true_in_ch != in_ch:
                logger.info(f"从瓶颈层恢复真实输入通道: {in_ch} → {true_in_ch}")
                in_ch = true_in_ch

        self.verifiers = {}
        for subj, sd in ckpt["verifier_states"].items():
            v = RSSICNNBinaryClassifier(in_ch, cfg or CNNConfig())
            v.load_state_dict(sd)
            v.eval()
            self.verifiers[subj] = v  # 保存在 CPU, 推理时按需移至 GPU

    @torch.no_grad()
    def predict_authentication(self, x, claimed, batch_size: int = 256):
        """分批推理, 模型按需移至 GPU, 推理后放回 CPU 释放显存。"""
        v = self.verifiers[claimed]
        v.to(self.device)
        try:
            n = len(x)
            scores = np.empty(n, dtype=np.float32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xt = prepare_cnn_input(x[start:end], self.device)
                scores[start:end] = v.predict_proba(xt)
        finally:
            v.cpu()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        accept_rate = np.mean(scores >= self.thresholds[claimed])
        return (accept_rate >= 0.5, float(np.mean(scores)), scores)
