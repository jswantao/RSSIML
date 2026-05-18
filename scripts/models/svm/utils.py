# -*- coding: utf-8 -*-
"""SVM 工具函数。"""
import logging
import warnings
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.exceptions import ConvergenceWarning

from scripts.models.base import compute_threshold

logger = logging.getLogger(__name__)


def svm_scores(verifier, x: np.ndarray) -> np.ndarray:
    """计算 SVM 验证器的认证分数。

    对特征矩阵的每一行返回 [0, 1] 之间的分数,
    表示该样本被接受为所属类别的概率。
    """
    raw = verifier.decision_function(x).astype(np.float64)
    np.clip(raw, -30.0, 30.0, out=raw)
    return 1.0 / (1.0 + np.exp(-raw))


def _train_single_verifier(idx, subject, y_enc, x, cfg):
    """单个用户 SVM 验证器训练 — 内存优化版。"""
    yb = (y_enc == idx).astype(np.int32)
    n_pos = int(np.sum(yb))
    n_neg = len(yb) - n_pos

    if n_pos < 2:
        return subject, None, 0.5, {"n_positive": n_pos, "skipped": True}

    rng = np.random.default_rng(cfg.random_seed + idx)
    max_neg = max(2000, n_pos * 10)

    if n_neg > max_neg:
        pos_idx = np.where(yb == 1)[0]
        neg_idx = rng.choice(
            np.where(yb == 0)[0], size=max_neg, replace=False)
        use_idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(use_idx)
        x_tr = x[use_idx]
        y_tr = yb[use_idx]
        n_neg_used = max_neg
    else:
        x_tr = x
        y_tr = yb
        n_neg_used = n_neg

    n_feats = x_tr.shape[1]
    n_samples = x_tr.shape[0]
    # 维度/样本比 > 0.5 时切换线性核，避免 RBF 在小样本高维场景过拟合
    if n_feats > 500 or n_feats / max(n_samples, 1) > 0.5:
        v = LinearSVC(
            class_weight="balanced", dual=False, max_iter=10000, tol=1e-4,
            random_state=cfg.random_seed,
        )
    else:
        v = SVC(
            kernel="rbf", probability=False,
            C=cfg.svm_C, gamma=cfg.svm_gamma,
            class_weight="balanced", cache_size=200,
            random_state=cfg.random_seed,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        v.fit(x_tr, y_tr)

    pos = svm_scores(v, x_tr[y_tr == 1])
    if n_neg_used > 0:
        neg_sample_idx = rng.choice(
            n_neg_used, size=min(2000, n_neg_used), replace=False)
        neg = svm_scores(v, x_tr[y_tr == 0][neg_sample_idx])
    else:
        neg = np.array([0.0])

    threshold, tinfo = compute_threshold(
        pos, neg, cfg.threshold_method, cfg.distance_threshold_quantile)

    return subject, v, threshold, {
        "n_positive": n_pos, "n_negative": n_neg_used,
        "threshold": float(threshold), "threshold_method": cfg.threshold_method,
        "threshold_details": tinfo,
        "mean_positive_score": float(np.mean(pos)),
        "mean_negative_score": float(np.mean(neg)),
    }
