# -*- coding: utf-8 -*-
"""SVM 身份认证模型 — 在线/离线 SVM 二分类验证器。

支持两种模式:
  - 离线 SVM: scikit-learn SVC (RBF/Linear 核)
  - 在线 SVM: SGDClassifier (hinge loss, 增量学习)

逐用户训练 (Per-User Binary Verification):
  对每个注册用户训练一个独立的二分类器 (genuine=+1, impostor=-1),
  推理时计算 decision_function 分数与用户专属阈值比较。
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from scripts.config import PipelineConfig
from scripts.models.metrics import MetricsCalculator, evaluate_authentication

logger = logging.getLogger(__name__)


@dataclass
class SVMConfig:
    """SVM 训练配置。"""
    threshold_method: str = "youden"
    random_seed: int = 42
    cv_folds: int = 5
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"


@dataclass
class AuthenticationModel:
    """认证模型容器 — 保存所有用户的验证器和预处理组件。"""
    verifiers: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    pca_model: Any = None
    scaler_model: Any = None
    feature_config: dict[str, Any] | None = None
    feature_dim: int | None = None
    data_source: str = "rssi"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("模型已保存: %s", path)

    @classmethod
    def load(cls, path: Path) -> "AuthenticationModel":
        with path.open("rb") as f:
            return pickle.load(f)


def svm_scores(verifier: Any, features: np.ndarray) -> np.ndarray:
    """计算 SVM 验证器对特征矩阵的决策函数分数。

    Args:
        verifier: 已训练的 SVM 模型 (SVC 或 SGDClassifier)。
        features: (N, D) 特征矩阵。

    Returns:
        (N,) 决策函数分数。
    """
    if hasattr(verifier, "decision_function"):
        return verifier.decision_function(features).astype(np.float64)
    elif hasattr(verifier, "predict_proba"):
        proba = verifier.predict_proba(features)
        return proba[:, 1].astype(np.float64) if proba.shape[1] > 1 else proba[:, 0].astype(np.float64)
    raise ValueError("verifier 必须支持 decision_function 或 predict_proba")


def compute_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "youden",
) -> float:
    """计算最优决策阈值。

    Args:
        scores: 决策函数分数。
        labels: 真实标签 (1=genuine, -1/0=impostor)。
        method: 阈值方法 — "youden" / "eer" / "median"。

    Returns:
        最优阈值。
    """
    if method == "median":
        genuine_scores = scores[labels == 1]
        return float(np.median(genuine_scores)) if len(genuine_scores) > 0 else 0.0

    if method == "eer":
        return _compute_eer_threshold(scores, labels)

    # Youden's J (默认)
    return _compute_youden_threshold(scores, labels)


def _compute_youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Youden's J 阈值 — 最大化 TPR - FPR。"""
    sorted_scores = np.sort(np.unique(scores))
    best_j, best_t = -np.inf, 0.0

    genuine = labels == 1
    impostor = ~genuine
    n_genuine = max(1, np.sum(genuine))
    n_impostor = max(1, np.sum(impostor))

    for t in sorted_scores:
        tpr = np.sum(scores[genuine] >= t) / n_genuine
        fpr = np.sum(scores[impostor] >= t) / n_impostor
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t

    return float(best_t)


def _compute_eer_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER 阈值 — FAR = FRR 的交叉点。"""
    sorted_scores = np.sort(np.unique(scores))
    genuine = labels == 1
    impostor = ~genuine
    n_genuine = max(1, np.sum(genuine))
    n_impostor = max(1, np.sum(impostor))

    best_diff, best_t = np.inf, 0.0
    for t in sorted_scores:
        frr = np.sum(scores[genuine] < t) / n_genuine
        far = np.sum(scores[impostor] >= t) / n_impostor
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff, best_t = diff, t

    return float(best_t)


class SVMAuthenticationTrainer:
    """SVM 认证训练器 — 逐用户二分类验证。"""

    def __init__(
        self,
        config: SVMConfig,
        pipeline_config: PipelineConfig | None = None,
        use_online: bool = False,
        online_kernel: str = "linear",
    ):
        self.config = config
        self.pcfg = pipeline_config or PipelineConfig.from_root()
        self.use_online = use_online
        self.online_kernel = online_kernel

    def train(
        self,
        data_file: Path,
        model_path: Path | None = None,
        data_source: str = "rssi",
    ) -> dict[str, Any]:
        """训练认证模型。

        Args:
            data_file: 包含 x_train, y_train, x_test, y_test 的 pickle。
            model_path: 模型保存路径。
            data_source: 数据源标识。

        Returns:
            训练结果 dict (含 model, verifiers, thresholds, system_metrics)。
        """
        logger.info("加载训练数据: %s", data_file)
        with data_file.open("rb") as f:
            data = pickle.load(f)

        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=object)

        pca = data.get("pca_model") or data.get("pca")
        scaler = data.get("scaler_model") or data.get("scaler")
        feature_config = data.get("feature_config")
        if hasattr(feature_config, "to_dict"):
            feature_config = feature_config.to_dict()

        subjects = sorted(set(y_train))
        logger.info("训练 %d 个用户的验证器", len(subjects))

        verifiers: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        t0 = time.time()

        for subj in subjects:
            genuine_mask = y_train == subj
            impostor_mask = ~genuine_mask

            if np.sum(genuine_mask) < 2 or np.sum(impostor_mask) < 2:
                logger.warning("用户 %s 样本不足, 跳过", subj)
                continue

            binary_labels = np.where(genuine_mask, 1, -1)

            if self.use_online:
                clf = SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=1e-4,
                    random_state=self.config.random_seed,
                    max_iter=1000,
                    tol=1e-3,
                )
            else:
                clf = SVC(
                    kernel=self.config.kernel,
                    C=self.config.C,
                    gamma=self.config.gamma,
                    random_state=self.config.random_seed,
                    probability=False,
                )

            clf.fit(x_train, binary_labels)
            verifiers[subj] = clf

            # 计算阈值
            scores = clf.decision_function(x_train)
            thresholds[subj] = compute_threshold(
                scores, binary_labels, self.config.threshold_method
            )

        train_dur = time.time() - t0
        logger.info("训练完成: %d 用户, 耗时 %.1fs", len(verifiers), train_dur)

        # 构建认证模型
        model = AuthenticationModel(
            verifiers=verifiers,
            thresholds=thresholds,
            pca_model=pca,
            scaler_model=scaler,
            feature_config=feature_config,
            feature_dim=x_train.shape[1] if x_train.ndim == 2 else None,
            data_source=data_source,
        )

        # 评估
        system_metrics = evaluate_authentication(
            model, x_test, y_test, data.get("auth_test_meta"),
            threshold_method=self.config.threshold_method,
        )

        if model_path:
            model.save(model_path)

        return {
            "model": model,
            "verifiers": verifiers,
            "thresholds": thresholds,
            "subjects": subjects,
            "system_metrics": system_metrics,
            "training_duration": train_dur,
        }
