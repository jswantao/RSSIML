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
    """认证模型容器 — 保存所有用户的验证器和预处理组件。

    SVM / CNN 训练器均使用此类封装训练结果。
    verifiers 字典的 value 类型:
      - SVM: sklearn SVC / SGDClassifier
      - CNN: torch.nn.Module (RSSICNNBinaryClassifier)
    """
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
    """计算验证器对特征矩阵的评分 — 兼容 SVM 和 CNN。

    SVM:  使用 decision_function (返回决策超平面距离)。
    CNN:  使用 forward → sigmoid (返回 [0,1] 概率)。

    Args:
        verifier: 已训练的模型 (SVC / SGDClassifier / nn.Module)。
        features: (N, D) 或 (N, C, W) 特征矩阵。

    Returns:
        (N,) 浮点分数。
    """
    # sklearn SVM
    if hasattr(verifier, "decision_function"):
        return np.asarray(verifier.decision_function(features), dtype=np.float64)
    if hasattr(verifier, "predict_proba"):
        proba = verifier.predict_proba(features)
        return proba[:, 1].astype(np.float64) if proba.shape[1] > 1 else proba[:, 0].astype(np.float64)

    # PyTorch nn.Module (CNN)
    try:
        import torch
        if isinstance(verifier, torch.nn.Module):
            verifier.eval()
            x_t = torch.from_numpy(np.asarray(features, dtype=np.float32))
            param = next(verifier.parameters(), None)
            if param is not None:
                x_t = x_t.to(param.device)
            with torch.no_grad():
                logits = verifier(x_t).cpu().numpy().squeeze(-1)
            return (1.0 / (1.0 + np.exp(-logits.astype(np.float64))))
    except ImportError:
        pass

    raise TypeError(
        f"verifier 类型 {type(verifier).__name__} 不受支持: "
        f"需要 sklearn 的 decision_function/predict_proba 或 torch.nn.Module"
    )


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
    n_genuine = max(1, int(np.sum(genuine)))
    n_impostor = max(1, int(np.sum(impostor)))

    for t in sorted_scores:
        tpr = np.sum(scores[genuine] >= t) / n_genuine
        fpr = np.sum(scores[impostor] >= t) / n_impostor
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)

    return best_t


def _compute_eer_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER 阈值 — FAR = FRR 的交叉点。"""
    sorted_scores = np.sort(np.unique(scores))
    genuine = labels == 1
    impostor = ~genuine
    n_genuine = max(1, int(np.sum(genuine)))
    n_impostor = max(1, int(np.sum(impostor)))

    best_diff, best_t = np.inf, 0.0
    for t in sorted_scores:
        frr = np.sum(scores[genuine] < t) / n_genuine
        far = np.sum(scores[impostor] >= t) / n_impostor
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff, best_t = diff, float(t)

    return best_t


class SVMAuthenticationTrainer:
    """SVM 认证训练器 — 逐用户二分类验证。

    支持交叉验证阈值选择 (cv_folds > 1 时启用)。
    """

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

    def _create_classifier(self) -> Any:
        """创建 SVM 分类器实例。"""
        if self.use_online:
            return SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-4,
                random_state=self.config.random_seed,
                max_iter=1000,
                tol=1e-3,
            )
        return SVC(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=self.config.gamma,
            random_state=self.config.random_seed,
            probability=False,
        )

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
        logger.info("训练 %d 个用户的验证器 (online=%s)", len(subjects), self.use_online)

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

            clf = self._create_classifier()
            clf.fit(x_train, binary_labels)
            verifiers[subj] = clf

            # 计算阈值: 使用交叉验证评估 (cv_folds > 1) 或训练集评分
            if self.config.cv_folds > 1 and len(x_train) >= self.config.cv_folds * 2:
                threshold = self._cv_threshold(x_train, binary_labels, subj)
            else:
                scores = clf.decision_function(x_train)
                threshold = compute_threshold(
                    scores, binary_labels, self.config.threshold_method
                )
            thresholds[subj] = threshold

        train_dur = time.time() - t0
        logger.info("训练完成: %d 用户, 耗时 %.1fs", len(verifiers), train_dur)

        # 构建认证模型
        model = AuthenticationModel(
            verifiers=verifiers,
            thresholds=thresholds,
            pca_model=pca,
            scaler_model=scaler,
            feature_config=feature_config if isinstance(feature_config, dict) else None,
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

    def _cv_threshold(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        subject: str,
    ) -> float:
        """使用交叉验证选择最优阈值。"""
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        all_scores: list[float] = []
        all_labels: list[int] = []

        for train_idx, val_idx in skf.split(x, labels):
            clf = self._create_classifier()
            clf.fit(x[train_idx], labels[train_idx])
            val_scores = clf.decision_function(x[val_idx])
            all_scores.extend(val_scores.tolist())
            all_labels.extend(labels[val_idx].tolist())

        return compute_threshold(
            np.array(all_scores),
            np.array(all_labels),
            self.config.threshold_method,
        )
