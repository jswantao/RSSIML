# -*- coding: utf-8 -*-
"""在线 SVM 验证器 — 基于 SGD + 可选 RBF 核近似，支持增量学习。

提供两种模式:
  - linear: SGDClassifier(hinge) 直接在线性特征上训练, O(d) 内存
  - rbf:    RBFSampler (随机 Fourier 特征) 近似 RBF 核 → SGDClassifier,
            非线性决策边界 + 在线增量训练, 逼近批处理 SVC(RBF) 效果

核近似原理:
  RBFSampler 使用 Bochner 定理将 RBF 核 K(x,y)=exp(-γ||x-y||²)
  近似为随机 Fourier 特征的内积: z(x)·z(y) ≈ K(x,y)
  n_components 越大近似越精确 (默认 200, 2-3× 原始维度)

在线学习策略:
  - 初始训练: 分批 partial_fit, 每批 1000 样本, 遍历 3-5 个 epoch
  - 增量更新: 单次 partial_fit 在新数据上微调
"""

import logging
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from scripts.models.base import compute_threshold
from scripts.models.svm.utils import svm_scores

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = 1e-4
_DEFAULT_MAX_EPOCHS = 5
_DEFAULT_BATCH_SIZE = 1000
_DEFAULT_N_COMPONENTS = 200


class OnlineSVMVerifier:
    """在线 SVM 二分类验证器 — SGD + 可选核近似。

    支持 linear (线性) 和 rbf (核近似) 两种模式,
    与 SVC 保持相同的 decision_function 接口。

    Example:
        >>> # 线性
        >>> v = OnlineSVMVerifier(kernel='linear', random_seed=42)
        >>> v.fit(x_train, y_train)
        >>> # RBF 核近似
        >>> v2 = OnlineSVMVerifier(kernel='rbf', n_components=200, random_seed=42)
        >>> v2.fit(x_train, y_train)
        >>> scores = v2.decision_function(x_test)
    """

    def __init__(
        self,
        kernel: str = "linear",
        alpha: float = _DEFAULT_ALPHA,
        max_epochs: int = _DEFAULT_MAX_EPOCHS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        n_components: int = _DEFAULT_N_COMPONENTS,
        gamma: float | str = "scale",
        random_seed: int = 42,
    ):
        if kernel not in ("linear", "rbf"):
            raise ValueError(f"kernel 必须是 'linear' 或 'rbf': {kernel}")

        self.kernel = kernel
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.gamma = gamma
        self.random_seed = random_seed

        # 核近似 (仅 rbf 模式)
        self._sampler: RBFSampler | None = None
        if kernel == "rbf":
            self._sampler = RBFSampler(
                gamma=gamma, n_components=n_components,
                random_state=random_seed,
            )

        self.scaler = StandardScaler()
        self._clf: SGDClassifier | None = None
        self._classes = np.array([0, 1])
        self._fitted = False
        self._class_weight: dict | None = None

    # ── 内部 ────────────────────────────────────────────────────────────

    def _build_clf(self, y: np.ndarray | None = None):
        """构建 SGDClassifier, 根据标签预先计算类别权重。

        class_weight='balanced' 不兼容 partial_fit, 需预先计算为 dict。
        """
        cw = None
        if y is not None:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight("balanced", classes=classes, y=y)
            cw = {int(c): float(w) for c, w in zip(classes, weights)}
            self._class_weight = cw
        else:
            cw = self._class_weight
        self._clf = SGDClassifier(
            loss="hinge", penalty="l2", alpha=self.alpha,
            max_iter=1000, tol=1e-4, random_state=self.random_seed,
            class_weight=cw, n_jobs=-1,
        )

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """核近似 (若启用) → 标准化 → float32 (节省内存)。

        仅在 RBF 核近似时需要 float64 精度, linear 核全程 float32。
        """
        need_f64 = self._sampler is not None
        X = np.asarray(X, dtype=np.float64 if need_f64 else np.float32)
        if self._sampler is not None:
            X = self._sampler.fit_transform(X) if fit else self._sampler.transform(X)
            X = X.astype(np.float32, copy=False)  # 核近似完成后立即降精度
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X.astype(np.float32, copy=False)

    # ── 公开 API (与 SVC 兼容) ───────────────────────────────────────────

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineSVMVerifier":
        """增量训练 (在线更新) — 首次调用 fit 核近似 + scaler。"""
        y = np.asarray(y, dtype=np.int32)

        if not self._fitted:
            X = self._transform(X, fit=True)
            self._classes = np.unique(y)
            if self._clf is None:
                self._build_clf(y)
            self._fitted = True
        else:
            X = self._transform(X, fit=False)

        self._clf.partial_fit(X, y, classes=self._classes)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineSVMVerifier":
        """初始批量训练 — fit 核近似 + 多次 epoch 遍历数据。

        使用索引洗牌代替 sklearn.utils.shuffle(X, y),
        避免全量数组拷贝导致的 OOM。
        """
        y = np.asarray(y, dtype=np.int32)
        X = self._transform(X, fit=True)
        self._classes = np.unique(y)
        self._fitted = True

        # 构建带类别权重的分类器 (预先计算避免 partial_fit 不兼容)
        self._build_clf(y)

        n = X.shape[0]
        bs = min(self.batch_size, max(1, n // 4))

        for epoch in range(self.max_epochs):
            idx = np.random.default_rng(self.random_seed + epoch).permutation(n)
            for start in range(0, n, bs):
                end = min(start + bs, n)
                batch_idx = idx[start:end]
                self._clf.partial_fit(X[batch_idx], y[batch_idx], classes=self._classes)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """原始决策值 (与 SVC.decision_function 兼容)。"""
        if not self._fitted or self._clf is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 或 partial_fit()")
        X = self._transform(X, fit=False)
        return self._clf.decision_function(X).astype(np.float64)


def train_online_verifier(
    idx: int,
    subject: str,
    y_enc: np.ndarray,
    x: np.ndarray,
    kernel: str = "linear",
    alpha: float = _DEFAULT_ALPHA,
    max_epochs: int = _DEFAULT_MAX_EPOCHS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    n_components: int = _DEFAULT_N_COMPONENTS,
    gamma: float | str = "scale",
    random_seed: int = 42,
    threshold_method: str = "youden",
    distance_threshold_quantile: float = 0.5,
) -> tuple:
    """训练单个用户的在线 SVM 验证器。

    与 svm/utils.py 的 _train_single_verifier 保持相同的返回格式,
    便于在 SVMAuthenticationTrainer 中替换。

    Returns:
        (subject, verifier, threshold, info_dict)
    """
    yb = (y_enc == idx).astype(np.int32)
    n_pos = int(np.sum(yb))

    if n_pos < 2:
        return subject, None, 0.5, {"n_positive": n_pos, "skipped": True}

    rng = np.random.default_rng(random_seed + idx)
    n_neg = len(yb) - n_pos
    max_neg = max(5000, n_pos * 5)

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

    v = OnlineSVMVerifier(
        kernel=kernel, alpha=alpha, max_epochs=max_epochs,
        batch_size=batch_size, n_components=n_components,
        gamma=gamma, random_seed=random_seed,
    )
    v.fit(x_tr, y_tr)

    pos = svm_scores(v, x_tr[y_tr == 1])
    if n_neg_used > 0:
        neg_sample_idx = rng.choice(
            n_neg_used, size=min(2000, n_neg_used), replace=False)
        neg = svm_scores(v, x_tr[y_tr == 0][neg_sample_idx])
    else:
        neg = np.array([0.0])

    threshold, tinfo = compute_threshold(
        pos, neg, threshold_method, distance_threshold_quantile)

    return subject, v, threshold, {
        "n_positive": n_pos, "n_negative": n_neg_used,
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "threshold_details": tinfo,
        "mean_positive_score": float(np.mean(pos)),
        "mean_negative_score": float(np.mean(neg)),
        "model_type": f"online_svm_{kernel}",
    }
