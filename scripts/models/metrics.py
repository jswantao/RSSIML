# -*- coding: utf-8 -*-
"""认证评估指标 — FAR / FRR / HTER / EER / 准确率。

逐用户二分类验证评估:
  对每个用户分别计算 FAR (冒名接受率) 和 FRR (本人拒绝率),
  然后汇总为系统级 HTER (半总错误率) 和全局准确率。

兼容 SVM (decision_function) 和 CNN (forward → sigmoid) 两类验证器。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """训练过程异常。"""
    pass


class MetricsCalculator:
    """认证指标计算器。"""

    @staticmethod
    def compute_far_frr(
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float,
    ) -> tuple[float, float]:
        """计算 FAR 和 FRR。

        Args:
            genuine_scores: 合法用户的分数。
            impostor_scores: 冒名者的分数。
            threshold: 决策阈值。

        Returns:
            (FAR, FRR) 元组。
        """
        n_genuine = max(1, len(genuine_scores))
        n_impostor = max(1, len(impostor_scores))

        frr = float(np.sum(genuine_scores < threshold)) / n_genuine
        far = float(np.sum(impostor_scores >= threshold)) / n_impostor

        return far, frr

    @staticmethod
    def compute_hter(far: float, frr: float) -> float:
        """HTER = (FAR + FRR) / 2"""
        return (far + frr) / 2.0

    @staticmethod
    def compute_accuracy(
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float,
    ) -> float:
        """计算认证准确率。"""
        tp = np.sum(genuine_scores >= threshold)
        tn = np.sum(impostor_scores < threshold)
        total = len(genuine_scores) + len(impostor_scores)
        return float(tp + tn) / max(1, total)


def _verifier_scores(verifier: Any, x: np.ndarray) -> np.ndarray:
    """统一的验证器评分 — 兼容 SVM 和 CNN。

    SVM: 使用 decision_function 或 predict_proba。
    CNN (nn.Module): 使用 forward → sigmoid 概率。

    Args:
        verifier: SVM 或 CNN 模型。
        x: (N, D) 特征矩阵。

    Returns:
        (N,) 浮点分数。
    """
    # sklearn SVM
    if hasattr(verifier, "decision_function"):
        return np.asarray(verifier.decision_function(x), dtype=np.float64)
    if hasattr(verifier, "predict_proba"):
        proba = verifier.predict_proba(x)
        return proba[:, 1].astype(np.float64) if proba.shape[1] > 1 else proba[:, 0].astype(np.float64)

    # PyTorch nn.Module (CNN)
    try:
        import torch
        if isinstance(verifier, torch.nn.Module):
            verifier.eval()
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            if next(verifier.parameters(), None) is not None:
                x_t = x_t.to(next(verifier.parameters()).device)
            with torch.no_grad():
                logits = verifier(x_t).cpu().numpy().squeeze(-1)
            # sigmoid → 概率
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    except ImportError:
        pass

    raise TypeError(
        f"verifier 类型 {type(verifier).__name__} 不受支持: "
        f"需要 sklearn 的 decision_function/predict_proba 或 torch.nn.Module"
    )


def evaluate_authentication(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    auth_test_meta: list[dict[str, Any]] | None = None,
    threshold_method: str = "youden",
) -> dict[str, Any]:
    """评估认证模型的系统级和用户级性能。

    兼容 SVM (decision_function) 和 CNN (forward) 验证器。

    Args:
        model: AuthenticationModel 实例 (含 verifiers, thresholds)。
        x_test: 测试特征矩阵。
        y_test: 测试标签。
        auth_test_meta: 认证测试元数据 (含 claimed_identity, true_label)。
        threshold_method: 阈值方法。

    Returns:
        系统指标 dict。
    """
    verifiers = model.verifiers if hasattr(model, "verifiers") else {}
    thresholds = model.thresholds if hasattr(model, "thresholds") else {}

    if not verifiers:
        return {
            "mean_hter": 1.0, "mean_far": 1.0, "mean_frr": 1.0,
            "global_accuracy": 0.0, "global_f1": 0.0,
            "std_hter": 0.0, "std_far": 0.0, "std_frr": 0.0,
            "user_metrics": {},
        }

    calc = MetricsCalculator()
    user_metrics: dict[str, dict[str, Any]] = {}
    all_correct = 0
    all_total = 0

    subjects = sorted(verifiers.keys())

    # 如果有认证测试元数据, 使用 claimed_identity + true_label
    if auth_test_meta and len(auth_test_meta) == len(y_test):
        for subj in subjects:
            if subj not in verifiers:
                continue
            verifier = verifiers[subj]
            threshold = thresholds.get(subj, 0.5)

            genuine_indices = []
            impostor_indices = []
            for i, meta in enumerate(auth_test_meta):
                claimed = meta.get("claimed_identity", y_test[i])
                true_label = meta.get("true_label", "genuine")
                if claimed == subj:
                    if true_label == "genuine":
                        genuine_indices.append(i)
                    else:
                        impostor_indices.append(i)

            if not genuine_indices and not impostor_indices:
                continue

            genuine_scores = _verifier_scores(
                verifier, x_test[genuine_indices]
            ) if genuine_indices else np.array([])

            impostor_scores = _verifier_scores(
                verifier, x_test[impostor_indices]
            ) if impostor_indices else np.array([])

            far, frr = calc.compute_far_frr(genuine_scores, impostor_scores, threshold)
            hter = calc.compute_hter(far, frr)

            user_metrics[subj] = {
                "far": far, "frr": frr, "hter": hter,
                "threshold": threshold,
                "n_genuine_tests": len(genuine_indices),
                "n_impostor_tests": len(impostor_indices),
            }

            all_correct += int(np.sum(genuine_scores >= threshold))
            all_correct += int(np.sum(impostor_scores < threshold))
            all_total += len(genuine_indices) + len(impostor_indices)

    else:
        # 无元数据: 使用 y_test 作为用户标签
        for subj in subjects:
            if subj not in verifiers:
                continue
            verifier = verifiers[subj]
            threshold = thresholds.get(subj, 0.5)

            genuine_mask = y_test == subj
            impostor_mask = ~genuine_mask

            if not np.any(genuine_mask):
                continue

            genuine_scores = _verifier_scores(verifier, x_test[genuine_mask])
            impostor_scores = _verifier_scores(verifier, x_test[impostor_mask])

            far, frr = calc.compute_far_frr(genuine_scores, impostor_scores, threshold)
            hter = calc.compute_hter(far, frr)

            user_metrics[subj] = {
                "far": far, "frr": frr, "hter": hter,
                "threshold": threshold,
                "n_genuine_tests": int(np.sum(genuine_mask)),
                "n_impostor_tests": int(np.sum(impostor_mask)),
            }

            all_correct += int(np.sum(genuine_scores >= threshold))
            all_correct += int(np.sum(impostor_scores < threshold))
            all_total += len(genuine_scores) + len(impostor_scores)

    # 系统级汇总
    if user_metrics:
        fars = [m["far"] for m in user_metrics.values()]
        frrs = [m["frr"] for m in user_metrics.values()]
        hters = [m["hter"] for m in user_metrics.values()]
    else:
        fars = frrs = hters = [1.0]

    global_acc = all_correct / max(1, all_total)

    mean_far = float(np.mean(fars))
    mean_frr = float(np.mean(frrs))
    precision = 1 - mean_far if mean_far < 1 else 0
    recall = 1 - mean_frr if mean_frr < 1 else 0
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "mean_hter": float(np.mean(hters)),
        "mean_far": mean_far,
        "mean_frr": mean_frr,
        "std_hter": float(np.std(hters)),
        "std_far": float(np.std(fars)),
        "std_frr": float(np.std(frrs)),
        "global_accuracy": global_acc,
        "global_f1": f1,
        "user_metrics": user_metrics,
    }
