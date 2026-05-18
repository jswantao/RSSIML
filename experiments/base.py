# -*- coding: utf-8 -*-
"""实验基础设施 — 共享的流水线构建、训练执行和切片评估。"""
from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from scripts.app_utils import (
    build_windows,
    extract_features_for_auth,
    find_processed_file,
)
from scripts.config import Defaults, PipelineConfig
from scripts.models import clear_gpu_memory, svm_scores
from scripts.pipeline_runner import AuthPipeline

logger = logging.getLogger(__name__)


class BaseExperimentRunner:
    """共享实验基础设施 — 流水线构建、SVM/CNN 训练、切片评估。"""

    def __init__(
        self,
        source: str,
        config: PipelineConfig | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ):
        self.source = source
        self._pcfg = config or PipelineConfig.from_root()
        self._check_cancelled = check_cancelled or (lambda: None)
        self.logger = logger
        self.results: dict[str, Any] = {}

    def _pipeline(self, **kw):
        defaults = dict(
            data_source=self.source, seed=Defaults.SEED, test_size=Defaults.TEST_SIZE,
            window_size=Defaults.WINDOW_SIZE, step_size=Defaults.STEP_SIZE, use_pca=False,
            use_cache=True, use_model_cache=False,
            cache_path=None,
            save_model=False, save_metrics=False, clean_intermediate=False,
        )
        if self.source == "csi":
            na = len(self._pcfg.csi_selected_actions) if self._pcfg.csi_selected_actions else 55
            defaults.setdefault("max_files_per_subject", na * 10)
            defaults.setdefault("use_online_svm", True)
            defaults.setdefault("online_kernel", "linear")
            defaults.setdefault("csi_denoise", "butterworth")
        defaults.update(kw)
        return AuthPipeline(**defaults)

    def _run_svm(self, **kw):
        p = self._pipeline(**kw)
        return p.run_svm()

    def _run_cnn(self, **kw):
        """训练 CNN — 分离流水线参数与训练超参数。

        流水线参数 (seed, use_model_cache, cache_path, ...) 传递给 AuthPipeline,
        CNN 超参数 (epochs, batch_size, lr, use_checkpoint, grad_accum,
        conv_channels, hidden_units) 传递给 run_cnn()。

        模型缓存 (use_model_cache=True):
          训练前检查 cache_path (或默认 cache/) 中是否已有同参数 CNN 结果。
          命中则直接返回缓存结果，跳过训练；未命中则训练完成后写入缓存。
        """
        cnn_kw = {}
        for k in ("epochs", "batch_size", "learning_rate",
                   "use_checkpoint", "gradient_accumulation_steps",
                   "conv_channels", "hidden_units"):
            if k in kw:
                cnn_kw[k] = kw.pop(k)
        # 显式提取缓存控制参数，确保传递至 AuthPipeline
        kw.setdefault("use_model_cache", True)
        p = self._pipeline(**kw)
        return p.run_cnn(
            epochs=cnn_kw.get("epochs", 15),
            batch_size=cnn_kw.get("batch_size", 64),
            learning_rate=cnn_kw.get("learning_rate", 0.001),
            use_checkpoint=cnn_kw.get("use_checkpoint", True),
            gradient_accumulation_steps=cnn_kw.get("gradient_accumulation_steps", 4),
            conv_channels=cnn_kw.get("conv_channels"),
            hidden_units=cnn_kw.get("hidden_units"),
            cancel_fn=self._check_cancelled,
        )

    def _compute_continuous_auth(self, result, smooth_window=Defaults.CONTINUOUS_AUTH_SMOOTH_WINDOW,
                                 subj=None):
        """持续认证评估 — 与推理阶段 `_render_continuous_auth` 完全一致。

        Args:
            result: SVM 训练结果 dict (含 model, verifiers, thresholds)。
            smooth_window: 滑动平均窗口, 默认 10。
            subj: 指定用户, None 则使用第一个 verifier。

        Returns:
            dict 含全部窗口级结果和汇总指标, 或 {}。
        """
        ctx = self._prepare_auth_context(result, subj=subj)
        if ctx is None:
            return {}

        subj, verifier, threshold, pca, scaler, fc, fd, raw = ctx

        
        windows = build_windows(raw)
        n = windows.shape[0]
        if n < 2:
            return {}

        try:
            # 特征提取 + 评分
            feats = extract_features_for_auth(
                windows, pca, scaler, fc, feature_dim=fd)
            all_scores = svm_scores(verifier, feats)

            # 前向滑动平均 
            smoothed = np.array([
                np.mean(all_scores[max(0, i - smooth_window + 1):i + 1])
                for i in range(n)
            ])
            decisions = [bool(s >= threshold) for s in smoothed]

            # 汇总
            longest = streak = 0
            for d in decisions:
                if d:
                    streak += 1
                    longest = max(longest, streak)
                else:
                    streak = 0
            switches = sum(1 for i in range(1, n)
                           if decisions[i] != decisions[i - 1])

            return {
                "用户": subj,
                "总窗口数": n,
                "接受率": float(np.mean(decisions)),
                "最终决策": "接受" if decisions[-1] else "拒绝",
                "最长连续接受": longest,
                "状态切换次数": switches,
                "平滑窗口": smooth_window,
                "阈值": float(threshold),
                "决策序列": [int(d) for d in decisions],
                "原始分数": all_scores.tolist() if len(all_scores) <= 2000
                            else all_scores[:2000].tolist(),
                "平滑分数": smoothed.tolist() if len(smoothed) <= 2000
                            else smoothed[:2000].tolist(),
            }
        except Exception as e:
            self.logger.warning(f"持续认证评估失败: {e}")
            return {}

    # ── 共享辅助方法 ───────────────────────────────────────────────────

    def _prepare_auth_context(self, result, subj=None):
        """准备认证上下文: 加载指定用户验证器、PCA/Scaler、原始数据。

        匹配用户标签到对应的 .mat 文件 (FXY→1, ...)。
        若 subj 未指定则使用第一个用户。

        Returns:
            (subj, verifier, threshold, pca, scaler, fc, fd, raw) | None
        """
        model = result.get("model")
        verifiers = result.get("verifiers", {})
        thresholds = result.get("thresholds", {})
        if not model or not verifiers:
            return None

        subjects = list(verifiers.keys())
        if not subjects:
            return None
        subj = subj or subjects[0]
        if subj not in verifiers:
            return None
        verifier = verifiers[subj]
        threshold = thresholds.get(subj, 0.5)

        pca = getattr(model, 'pca_model', None)
        scaler = getattr(model, 'scaler_model', None)
        fc = getattr(model, 'feature_config', None)
        fd = getattr(model, 'feature_dim', None)
        # 仅当 Scaler 缺失时才回退 (PCA 在 use_pca=False 时合法为 None)
        if scaler is None:
            pp = find_processed_file("rssi")
            if pp and pp.exists():
                with pp.open("rb") as pf:
                    proc = pickle.load(pf)
                pca = proc.get("pca_model") or proc.get("pca")
                scaler = proc.get("scaler_model") or proc.get("scaler")

        raw_dir = self._pcfg.raw_dir
        mat_files = sorted(raw_dir.glob("*.mat"))
        unmap = self._pcfg.subject_unmap("rssi")
        raw_subj = unmap.get(subj, subj)
        subj_files = [f for f in mat_files
                      if f.name.lower().startswith(f"wipin_{raw_subj.lower()}")]
        if not subj_files:
            subj_files = mat_files[:1]
        if not subj_files:
            return None

        try:
            mat = loadmat(subj_files[0])
            if "RSSI" not in mat:
                return None
            raw = np.asarray(mat["RSSI"], dtype=np.float32)
            return (subj, verifier, threshold, pca, scaler, fc, fd, raw)
        except Exception as e:
            self.logger.warning(f"加载原始数据失败 ({subj}): {e}")
            return None
