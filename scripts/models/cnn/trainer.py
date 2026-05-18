# -*- coding: utf-8 -*-
"""CNN 训练器与推理引擎。"""
from __future__ import annotations

import gc
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.config import PipelineConfig
from scripts.models.cnn.models import CNNConfig, RSSICNNBinaryClassifier, CNNAuthenticationModel
from scripts.models.metrics import evaluate_authentication, MetricsCalculator

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class CNNTrainConfig:
    """CNN 训练超参数。"""
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    threshold_method: str = "youden"
    random_seed: int = 42
    gradient_accumulation_steps: int = 1
    use_amp: bool = True


class CNNTrainer:
    """CNN 认证训练器。"""

    def __init__(
        self,
        model_cfg: CNNConfig | None = None,
        train_config: CNNTrainConfig | None = None,
        config: PipelineConfig | None = None,
        cancel_fn: Callable[[], None] | None = None,
    ):
        self.model_cfg = model_cfg or CNNConfig()
        self.train_config = train_config or CNNTrainConfig()
        self.pcfg = config or PipelineConfig.from_root()
        self.cancel_fn = cancel_fn or (lambda: None)

    def train_authentication(
        self,
        data_file: Path,
        model_path: Path | None = None,
    ) -> dict[str, Any]:
        """训练 CNN 认证模型。"""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch 不可用")

        logger.info("加载训练数据: %s", data_file)
        with data_file.open("rb") as f:
            data = pickle.load(f)

        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=object)

        subjects = sorted(set(y_train))
        tc = self.train_config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("CNN 训练: %d 用户, device=%s, epochs=%d, batch=%d",
                     len(subjects), device, tc.epochs, tc.batch_size)

        verifiers: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        all_models: dict[str, Path] = {}
        t0 = time.time()

        for subj in subjects:
            self.cancel_fn()

            genuine_mask = y_train == subj
            binary_labels = np.where(genuine_mask, 1, 0).astype(np.float32)

            # 准备数据
            x_tensor = torch.from_numpy(x_train).to(device)
            y_tensor = torch.from_numpy(binary_labels).unsqueeze(1).to(device)

            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True)

            # 创建模型
            n_channels = x_train.shape[1] if x_train.ndim == 3 else x_train.shape[-1]
            model = RSSICNNBinaryClassifier(
                input_channels=n_channels,
                num_classes=1,
                config=self.model_cfg,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay,
            )
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler("cuda") if tc.use_amp and device.type == "cuda" else None

            # 训练循环
            model.train()
            for epoch in range(tc.epochs):
                self.cancel_fn()
                epoch_loss = 0.0
                n_batches = 0

                for batch_idx, (xb, yb) in enumerate(loader):
                    if tc.use_amp and scaler is not None:
                        with torch.amp.autocast("cuda"):
                            logits = model(xb)
                            loss = criterion(logits, yb) / tc.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % tc.gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        logits = model(xb)
                        loss = criterion(logits, yb) / tc.gradient_accumulation_steps
                        loss.backward()
                        if (batch_idx + 1) % tc.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    epoch_loss += loss.item() * tc.gradient_accumulation_steps
                    n_batches += 1

                if epoch % 5 == 0 or epoch == tc.epochs - 1:
                    logger.info("  用户 %s epoch %d/%d loss=%.4f",
                                subj, epoch + 1, tc.epochs, epoch_loss / max(1, n_batches))

            verifiers[subj] = model
            # 计算阈值
            model.eval()
            with torch.no_grad():
                all_logits = model(x_tensor).cpu().numpy().squeeze()
                all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
            thresholds[subj] = float(np.median(all_probs[genuine_mask]))

        train_dur = time.time() - t0
        logger.info("CNN 训练完成: %d 用户, 耗时 %.1fs", len(verifiers), train_dur)

        # 保存检查点
        checkpoint_path = ""
        if model_path:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "subjects": subjects,
                "thresholds": thresholds,
                "model_cfg": self.model_cfg,
            }, model_path)
            checkpoint_path = str(model_path)

        # 评估
        from scripts.models.svm import AuthenticationModel
        auth_model = AuthenticationModel(
            verifiers=verifiers,
            thresholds=thresholds,
        )

        system_metrics = evaluate_authentication(
            auth_model, x_test, y_test,
            threshold_method=tc.threshold_method,
        )

        return {
            "model": auth_model,
            "verifiers": verifiers,
            "thresholds": thresholds,
            "subjects": subjects,
            "system_metrics": system_metrics,
            "training_duration": train_dur,
            "checkpoint_path": checkpoint_path,
        }


class CNNInference:
    """CNN 推理引擎 — 从检查点加载模型并执行推理。"""

    def __init__(self, model_path: Path | str):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch 不可用")
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint = None
        self._models: dict[str, RSSICNNBinaryClassifier] = {}

    def load(self) -> None:
        """加载检查点。"""
        self._checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        logger.info("CNN 检查点已加载: %s", self.model_path)

    def predict(self, features: np.ndarray, subject: str) -> np.ndarray:
        """对指定用户执行推理。

        Args:
            features: (N, C, W) 或 (N, D) 特征。
            subject: 待验证的用户 ID。

        Returns:
            (N,) 概率分数。
        """
        if self._checkpoint is None:
            self.load()

        if subject not in self._models:
            logger.warning("用户 %s 不在检查点中", subject)
            return np.zeros(features.shape[0])

        model = self._models[subject]
        model.eval()

        x_tensor = torch.from_numpy(features.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = model(x_tensor).cpu().numpy().squeeze()

        return 1 / (1 + np.exp(-logits))  # sigmoid

    @property
    def subjects(self) -> list[str]:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get("subjects", [])

    @property
    def thresholds(self) -> dict[str, float]:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get("thresholds", {})
