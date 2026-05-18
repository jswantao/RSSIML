# -*- coding: utf-8 -*-
"""认证评估指标 — FAR / FRR / HTER / EER / 准确率。

逐用户二分类验证评估:
  对每个用户分别计算 FAR (冒名接受率) 和 FRR (本人拒绝率),
  然后汇总为系统级 HTER (半总错误率) 和全局准确率。
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


def evaluate_authentication(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    auth_test_meta: list[dict[str, Any]] | None = None,
    threshold_method: str = "youden",
) -> dict[str, Any]:
    """评估认证模型的系统级和用户级性能。

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
        return {"mean_hter": 1.0, "mean_far": 1.0, "mean_frr": 1.0, "global_accuracy": 0.0}

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

            genuine_scores = np.array([
                verifier.decision_function(x_test[i:i+1])[0]
                for i in genuine_indices
            ]) if genuine_indices else np.array([])

            impostor_scores = np.array([
                verifier.decision_function(x_test[i:i+1])[0]
                for i in impostor_indices
            ]) if impostor_indices else np.array([])

            far, frr = calc.compute_far_frr(genuine_scores, impostor_scores, threshold)
            hter = calc.compute_hter(far, frr)

            user_metrics[subj] = {
                "far": far, "frr": frr, "hter": hter,
                "threshold": threshold,
                "n_genuine_tests": len(genuine_indices),
                "n_impostor_tests": len(impostor_indices),
            }

            # 全局准确率
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

            genuine_scores = verifier.decision_function(x_test[genuine_mask])
            impostor_scores = verifier.decision_function(x_test[impostor_mask])

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

    # F1 近似
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
