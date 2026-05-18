# -*- coding: utf-8 -*-
"""基础训练工具 — 指标计算、阈值、认证评估。"""
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
logger = logging.getLogger(__name__)
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """评估指标计算。"""
    # ... (保持原代码不变)
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        pos_label: str = "known",
    ) -> dict:
        if average == "binary":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0)),
                "recall": float(recall_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0)),
                "f1": float(f1_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0)),
            }
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(
                y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(
                y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(
                y_true, y_pred, average=average, zero_division=0)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 共享: 阈值计算 (保持不变)
# ══════════════════════════════════════════════════════════════════════════════

_THRESHOLD_SEARCH_POINTS = 100
_EER_SEARCH_POINTS = 200


def compute_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    method: str = "youden",
    quantile: float = 0.95,
) -> tuple[float, dict]:
    # ... (保持原代码不变)
    if len(pos_scores) == 0:
        return 0.5, {"method": "fallback", "reason": "no_positive_samples"}

    if method == "fixed":
        return 0.5, {"method": "fixed"}

    if method == "quantile":
        t = float(np.quantile(pos_scores, 1.0 - quantile))
        return t, {"method": "quantile", "quantile": quantile}

    if method == "eer":
        return _compute_eer_threshold(pos_scores, neg_scores)

    best_t, best_j = 0.5, -1.0
    # 向量化: 广播比较替代逐阈值循环
    thresholds = np.linspace(0.01, 0.99, _THRESHOLD_SEARCH_POINTS)
    tpr = np.mean(pos_scores[:, np.newaxis] >= thresholds, axis=0)
    fpr = np.mean(neg_scores[:, np.newaxis] >= thresholds, axis=0)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_t = float(thresholds[best_idx])
    best_j = float(j_scores[best_idx])

    if best_j < 0.1:
        fallback = float(np.median(pos_scores))
        return fallback, {
            "method": "youden_fallback",
            "youden_index": float(best_j),
            "positive_median": fallback,
        }

    return float(best_t), {"method": "youden", "youden_index": float(best_j)}


def _compute_eer_threshold(pos_scores, neg_scores):
    thresholds = np.linspace(0.0, 1.0, _EER_SEARCH_POINTS)
    best_t, best_diff = 0.5, 1.0
    
    # 向量化计算
    far_rates = np.mean(neg_scores[:, np.newaxis] >= thresholds, axis=0)
    frr_rates = np.mean(pos_scores[:, np.newaxis] < thresholds, axis=0)
    diffs = np.abs(far_rates - frr_rates)
    
    best_idx = np.argmin(diffs)
    best_t = float(thresholds[best_idx])
    far_at_t = float(far_rates[best_idx])
    frr_at_t = float(frr_rates[best_idx])
    
    return best_t, {
        "method": "eer",
        "eer": float((far_at_t + frr_at_t) / 2),
        "far_at_eer": far_at_t,
        "frr_at_eer": frr_at_t,
    }


def evaluate_authentication(
    verifier_predict_fn,
    thresholds: dict[str, float],
    subjects: list[str],
    x_test: np.ndarray,
    y_test_raw: np.ndarray,
) -> dict:
    # ... (保持原代码不变)
    user_metrics, all_preds, all_labels = {}, [], []

    for subj in subjects:
        if subj not in thresholds:
            logger.warning(f"用户 {subj} 无阈值，跳过评估")
            continue

        scores = verifier_predict_fn(subj, x_test)
        preds = (scores >= thresholds[subj]).astype(bool)
        genuine = (y_test_raw == subj)

        n_g = int(np.count_nonzero(genuine))
        n_i = int(np.count_nonzero(~genuine))
        frr = float(1.0 - np.mean(preds[genuine])) if n_g else 0.0
        far = float(np.mean(preds[~genuine])) if n_i else 0.0

        user_metrics[subj] = {
            "far": far,
            "frr": frr,
            "hter": (far + frr) / 2.0,
            "threshold": float(thresholds[subj]),
            "n_genuine_tests": n_g,
            "n_impostor_tests": n_i,
        }
        all_preds.extend(preds.tolist())
        all_labels.extend(genuine.tolist())

    if not user_metrics:
        return {
            "mean_far": 0.0, "std_far": 0.0,
            "mean_frr": 0.0, "std_frr": 0.0,
            "mean_hter": 0.0, "std_hter": 0.0,
            "worst_hter": 0.0, "best_hter": 0.0,
            "global_accuracy": 0.0, "global_f1": 0.0,
            "user_metrics": {},
        }

    fars = [m["far"] for m in user_metrics.values()]
    frrs = [m["frr"] for m in user_metrics.values()]
    hters = [m["hter"] for m in user_metrics.values()]
    ap, al = np.array(all_preds), np.array(all_labels)

    return {
        "mean_far": float(np.mean(fars)),
        "std_far": float(np.std(fars)),
        "mean_frr": float(np.mean(frrs)),
        "std_frr": float(np.std(frrs)),
        "mean_hter": float(np.mean(hters)),
        "std_hter": float(np.std(hters)),
        "worst_hter": float(np.max(hters)),
        "best_hter": float(np.min(hters)),
        "global_accuracy": float(accuracy_score(al, ap)),
        "global_f1": float(f1_score(al, ap, zero_division=0)),
        "user_metrics": user_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
