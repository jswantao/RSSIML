# -*- coding: utf-8 -*-
"""CNN 训练器与推理引擎。

训练: 逐用户训练二分类 CNN → 保存模型权重检查点。
推理: 加载检查点 → 对指定用户执行 sigmoid 概率预测。
"""
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
from scripts.models.cnn.models import RSSICNNBinaryClassifier, CNNAuthenticationModel
from scripts.models.cnn.models import CNNConfig
from scripts.models.config import CNNTrainConfig
from scripts.models.base import evaluate_authentication

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class CNNTrainer:
    """CNN 认证训练器 — 逐用户 1D-CNN 二分类。

    训练流程:
      1. 加载窗口数据 (N, C, W)
      2. 对每个用户: genuine=1, impostor=0 → BCEWithLogitsLoss
      3. 训练完成后保存模型权重到检查点
      4. 使用 evaluate_authentication 评估 (兼容 nn.Module)
    """

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
        """训练 CNN 认证模型。

        Args:
            data_file: pickle 文件 (含 x_train, y_train, x_test, y_test)。
                       x_train shape = (N, C, W) 三维窗口数据。
            model_path: 检查点保存路径 (.pt)。

        Returns:
            训练结果 dict。
        """
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

        # 输入维度: (N, C, W) → C = n_channels
        if x_train.ndim == 3:
            n_channels = x_train.shape[1]
        elif x_train.ndim == 2:
            n_channels = x_train.shape[1]
        else:
            raise ValueError(f"x_train 维度异常: {x_train.shape}")

        logger.info(
            "CNN 训练: %d 用户, device=%s, epochs=%d, batch=%d, channels=%d",
            len(subjects), device, tc.epochs, tc.batch_size, n_channels,
        )

        # 将全量数据一次性移到设备上 (避免逐用户重复传输)
        x_tensor = torch.from_numpy(x_train).to(device)

        verifiers: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        state_dicts: dict[str, dict] = {}
        t0 = time.time()

        for subj in subjects:
            self.cancel_fn()

            genuine_mask = y_train == subj
            binary_labels = np.where(genuine_mask, 1, 0).astype(np.float32)
            y_tensor = torch.from_numpy(binary_labels).unsqueeze(1).to(device)

            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True)

            model = RSSICNNBinaryClassifier(
                input_channels=n_channels,
                num_classes=1,
                config=self.model_cfg,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay,
            )
            criterion = nn.BCEWithLogitsLoss()
            scaler = (
                torch.amp.GradScaler("cuda")
                if tc.use_amp and device.type == "cuda"
                else None
            )

            model.train()
            for epoch in range(tc.epochs):
                self.cancel_fn()
                epoch_loss = 0.0
                n_batches = 0

                for batch_idx, (xb, yb) in enumerate(loader):
                    if scaler is not None:
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
                    logger.info(
                        "  用户 %s epoch %d/%d loss=%.4f",
                        subj, epoch + 1, tc.epochs,
                        epoch_loss / max(1, n_batches),
                    )

            model.eval()
            verifiers[subj] = model
            state_dicts[subj] = model.state_dict()

            # 计算阈值: 对训练数据执行推理取 genuine 概率中位数
            with torch.no_grad():
                all_logits = model(x_tensor).cpu().numpy().squeeze(-1)
                all_probs = 1.0 / (1.0 + np.exp(-all_logits))
            thresholds[subj] = float(np.median(all_probs[genuine_mask]))

        train_dur = time.time() - t0
        logger.info("CNN 训练完成: %d 用户, 耗时 %.1fs", len(verifiers), train_dur)

        # 保存检查点 (包含模型权重)
        checkpoint_path = ""
        if model_path:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "subjects": subjects,
                "thresholds": thresholds,
                "model_cfg": self.model_cfg,
                "input_channels": n_channels,
                "state_dicts": state_dicts,
            }, model_path)
            checkpoint_path = str(model_path)
            logger.info("CNN 检查点已保存: %s", model_path)

        # 构建认证模型
        from scripts.models.svm import AuthenticationModel
        auth_model = AuthenticationModel(
            verifiers=verifiers,
            thresholds=thresholds,
        )

        # 评估 (evaluate_authentication 已兼容 nn.Module verifier)
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
