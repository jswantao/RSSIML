# -*- coding: utf-8 -*-
"""SVM 训练器 — 身份认证。"""
import gc
import json
import logging
import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

from scripts.config import PipelineConfig
from scripts.models.config import SVMConfig
from scripts.models.base import (
    MetricsCalculator, compute_threshold, evaluate_authentication,
)
from scripts.models.exceptions import TrainingError
from scripts.models.svm.model import AuthenticationModel
from scripts.models.svm.utils import svm_scores, _train_single_verifier
from scripts.models.memory import get_memory_monitor

_memory_monitor = get_memory_monitor()
logger = logging.getLogger(__name__)
from scripts.models import MODEL_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# SVM 认证训练器
# ══════════════════════════════════════════════════════════════════════════════

class SVMAuthenticationTrainer:
    """SVM 身份认证训练器 — 支持批处理 (RBF SVC) 和在线 (SGD) 两种模式。"""

    def __init__(
        self,
        config: SVMConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        use_online: bool = False,
        online_kernel: str = "linear",
    ) -> None:
        self.pcfg = pipeline_config or PipelineConfig.from_root()
        self.cfg = config or SVMConfig()
        self.encoder = LabelEncoder()
        self.use_online = use_online
        self.online_kernel = online_kernel

    def train(self, data_file: Path | None = None, model_path: Path | None = None,
              pca_model=None, scaler_model=None, data_source: str | None = None) -> dict:
        data = self._load(data_file)
        pca = pca_model or data.get("pca_model") or data.get("pca")
        scaler = scaler_model or data.get("scaler_model") or data.get("scaler")
        ds = data_source or data.get("data_source")
        try:
            return self._train_core(
                np.asarray(data["x_train"], dtype=np.float32),
                np.asarray(data["y_train"], dtype=object),
                np.asarray(data["x_test"], dtype=np.float32),
                np.asarray(data["y_test"], dtype=object),
                self._extract_feature_config(data),
                model_path,
                pca, scaler, ds,
            )
        finally:
            del data

    def train_from_arrays(
        self, x_tr: np.ndarray, y_tr_raw: np.ndarray,
        x_te: np.ndarray, y_te_raw: np.ndarray,
        feature_config: dict | None = None,
        model_path: Path | None = None,
        pca_model=None, scaler_model=None,
        data_source: str | None = None,
    ) -> dict:
        """直接从内存数组训练, 跳过文件序列化往返, 避免 OOM。"""
        return self._train_core(
            np.asarray(x_tr, dtype=np.float32),
            np.asarray(y_tr_raw, dtype=object),
            np.asarray(x_te, dtype=np.float32),
            np.asarray(y_te_raw, dtype=object),
            feature_config,
            model_path,
            pca_model, scaler_model, data_source,
        )

    def _extract_feature_config(self, data: dict) -> dict | None:
        if "meta" in data:
            meta = data["meta"]
            return {
                "feature_groups": meta.get("feature_groups", ["spectral", "statistical"]),
                "low_freq_bins": meta.get("low_freq_bins", 16),
                "denoise": meta.get("denoise"),
                "denoise_kernel": meta.get("denoise_kernel", 5),
            }
        if self.pcfg is not None:
            return {
                "feature_groups": list(self.pcfg.__dict__.get("feature_groups",
                    ("spectral", "statistical", "temporal"))),
                "low_freq_bins": getattr(self.pcfg, "low_freq_bins", 16),
                "denoise": getattr(self.pcfg, "denoise", None),
                "denoise_kernel": getattr(self.pcfg, "denoise_kernel", 5),
            }
        return None

    def _train_core(
        self, x_tr: np.ndarray, y_tr_raw: np.ndarray,
        x_te: np.ndarray, y_te_raw: np.ndarray,
        feature_config: dict | None,
        model_path: Path | None,
        pca_model=None, scaler_model=None,
        data_source: str | None = None,
    ) -> dict:
        y_tr_enc = self.encoder.fit_transform(y_tr_raw).astype(np.int64)
        subjects = [str(c) for c in self.encoder.classes_]

        if len(subjects) < 2:
            raise TrainingError(f"需要至少 2 个用户: 当前 {len(subjects)}")

        logger.info(
            f"SVM 认证: {len(subjects)} 用户, "
            f"{x_tr.shape[0]} 训练, 特征 {x_tr.shape[1]}D")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        verifiers, thresholds, user_train = {}, {}, {}
        t0 = time.time()
        # 根据数据规模估算每线程内存, 限制并行度防止 OOM
        n_mb = x_tr.shape[0] * x_tr.shape[1] * 4 / (1024**2)  # float32 MiB
        if self.use_online:
            est_per_worker_mb = n_mb * 2.5  # use_idx副本 + float64变换 → ~2.5×
        else:
            est_per_worker_mb = n_mb * 1.2  # SVC 内存开销较小
        mem_safe_workers = max(1, int(4000 / max(est_per_worker_mb, 1)))  # 4GiB 上限
        max_workers = min(mem_safe_workers, 8, len(subjects))
        if _memory_monitor.should_degrade():
            max_workers = 1
        model_label = f"在线SVM({self.online_kernel})" if self.use_online else "SVM"

        logger.info(f"并行训练 {len(subjects)} 个用户验证器 ({model_label}, workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            if self.use_online:
                from scripts.models.svm.online import train_online_verifier
                futures = {
                    ex.submit(
                        train_online_verifier, i, subjects[i], y_tr_enc, x_tr,
                        kernel=self.online_kernel,
                        random_seed=self.cfg.random_seed,
                        threshold_method=self.cfg.threshold_method,
                        distance_threshold_quantile=self.cfg.distance_threshold_quantile,
                    ): i
                    for i in range(len(subjects))
                }
            else:
                futures = {
                    ex.submit(_train_single_verifier, i, subjects[i], y_tr_enc, x_tr, self.cfg): i
                    for i in range(len(subjects))
                }
            for fut in as_completed(futures):
                try:
                    subj_name, verifier, threshold, info = fut.result()
                except Exception as exc:
                    logger.warning("验证器训练失败 (用户 %s): %s",
                                   subjects[futures[fut]], exc)
                    continue
                if verifier is not None:
                    verifiers[subj_name] = verifier
                    thresholds[subj_name] = threshold
                    user_train[subj_name] = info

        elapsed = time.time() - t0
        logger.info(f"验证器训练完成 ({elapsed:.1f}s)")

        def predict_fn(subj, x):
            return svm_scores(verifiers[subj], x)

        test_metrics = evaluate_authentication(
            predict_fn, thresholds, subjects, x_te, y_te_raw)

        model = AuthenticationModel(
            verifiers, thresholds, self.encoder, subjects,
            self.cfg.threshold_method,
        )
        if feature_config is not None:
            model.feature_config = feature_config
        if pca_model is not None:
            model.pca_model = pca_model
        if scaler_model is not None:
            model.scaler_model = scaler_model
        if data_source is not None:
            model.data_source = data_source
        model.feature_dim = x_tr.shape[1]  # 存储训练特征维度供推理校验
        self._save(model, test_metrics, model_path)

        logger.info(
            f"SVM 认证完成: HTER={test_metrics['mean_hter']:.4f}, "
            f"FAR={test_metrics['mean_far']:.4f}")

        return {
            "model": model,
            "verifiers": verifiers,
            "thresholds": thresholds,
            "subjects": subjects,
            "user_train_metrics": user_train,
            "system_metrics": test_metrics,
        }

    def _load(self, path: Path | None) -> dict:
        path = path or (self.pcfg.data_dir / "rssi_processed_authentication.pkl")
        with path.open("rb") as f:
            return pickle.load(f)

    def _save(self, model, metrics, model_path: Path | None = None):
        self.pcfg.model_dir.mkdir(parents=True, exist_ok=True)
        self.pcfg.result_dir.mkdir(parents=True, exist_ok=True)
        path = model_path or (self.pcfg.model_dir / "svm_authentication.pkl")
        with path.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with (self.pcfg.result_dir / "authentication_metrics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({
                "model": "SVM Per-User Binary Verification",
                "model_version": MODEL_VERSION,
                "task_type": "authentication",
                "threshold_method": model.threshold_method,
                "num_users": len(model.subjects),
                "users": model.subjects,
                "system_metrics": {
                    k: v for k, v in metrics.items()
                    if k != "user_metrics"
                },
                "user_metrics": metrics.get("user_metrics", {}),
            }, f, indent=2, ensure_ascii=False)
