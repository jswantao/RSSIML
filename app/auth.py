# -*- coding: utf-8 -*-
"""认证工具函数 — 模型加载/路径、CSI 拼接、用户映射、分数图表。"""
from __future__ import annotations

import pickle
import re
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import loadmat

from scripts.app_utils import (
    FONT_SIZES as _FS,
    build_windows as _build_windows,
    extract_features_for_auth as _extract_features_for_auth,
    find_processed_file as _find_processed_file,
    setup_paper_style as _setup_paper_style,
)
from scripts.build_sliding_windows import WindowBuilder, WindowConfig
from scripts.config import PipelineConfig
from scripts.models import CNNInference, svm_scores
from scripts.pipeline_runner import AuthPipeline

_CONFIG = PipelineConfig.from_root()


def get_model_path(source: str, model_type: str) -> Path:
    suffix = "_npy" if source == "csi" else ""
    if model_type == "svm":
        return _CONFIG.model_dir / f"svm_authentication{suffix}.pkl"
    return _CONFIG.model_dir / f"cnn_authentication{suffix}.pt"


def load_svm_model(source: str):
    path = get_model_path(source, "svm")
    if not path.exists():
        return None, None, None
    with path.open("rb") as f:
        model = pickle.load(f)
    pca = getattr(model, "pca_model", None)
    scaler = getattr(model, "scaler_model", None)
    return model, pca, scaler


def load_cnn_model(source: str):
    path = get_model_path(source, "cnn")
    if not path.exists():
        return None
    try:
        return CNNInference(path)
    except Exception:
        return None


def normalize_model_keys(model, source: str) -> dict:
    """将模型中的用户键统一为标准格式。"""
    subjects = list(model.verifiers.keys())
    map_disk2ui = {}
    if source == "csi":
        from scripts.config import PipelineConfig
        map_disk2ui = PipelineConfig.SUBJECT_MAP_CSI
    elif source == "rssi":
        from scripts.config import PipelineConfig
        map_disk2ui = PipelineConfig.SUBJECT_MAP_RSSI
    if not map_disk2ui:
        return {k: model.verifiers[k] for k in subjects}
    mapped_verifiers = {}
    for subj_id in subjects:
        unif_id = map_disk2ui.get(str(subj_id), str(subj_id))
        mapped_verifiers[str(unif_id)] = model.verifiers[subj_id]
    model.verifiers = mapped_verifiers
    if hasattr(model, "thresholds"):
        mapped_thresholds = {}
        for subj_id, thr in model.thresholds.items():
            unif_id = map_disk2ui.get(str(subj_id), str(subj_id))
            mapped_thresholds[str(unif_id)] = thr
        model.thresholds = mapped_thresholds
    return mapped_verifiers


def get_available_subjects(svm_model, cnn_model, source: str) -> list[str]:
    model = svm_model or cnn_model
    if model is None:
        return [str(i) for i in range(1, 6 if source == "rssi" else 20)]
    subjects = list(getattr(model, "subjects", []) or
                    getattr(model, "verifiers", {}).keys())
    return sorted(subjects, key=int)


def combine_csi_files(data_dir: Path, subject: str, trial: int = 0,
                      max_files: int | None = None) -> np.ndarray | None:
    """将同一用户多个动作的 CSI 文件沿时间轴拼接为连续样本。"""
    from scripts.data_loader import load_npy_matrix
    disk_subj = _CONFIG.subject_unmap("csi").get(subject, subject)
    pattern = re.compile(rf"^{disk_subj}_(\d+)_(\d+)\.npy$")
    by_action: dict[int, list[Path]] = {}
    for fp in data_dir.glob(f"{disk_subj}_*.npy"):
        m = pattern.match(fp.name)
        if not m:
            continue
        by_action.setdefault(int(m.group(1)), []).append((int(m.group(2)), fp))
    if not by_action:
        return None
    selected = []
    for action in sorted(by_action):
        files = sorted(by_action[action], key=lambda x: x[0])
        if trial > 0 and trial <= len(files):
            selected.append(files[trial - 1][1])
        elif files:
            selected.append(files[0][1])
    if max_files:
        selected = selected[:max_files]
    if not selected:
        return None
    arrays = []
    for fp in selected:
        try:
            arrays.append(load_npy_matrix(fp))
        except (ValueError, OSError):
            continue
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0).astype(np.float32)


def load_uploaded(uploaded, source: str) -> np.ndarray:
    if source == "rssi":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        mat = loadmat(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        if "RSSI" not in mat:
            raise ValueError("MAT 文件缺少 RSSI 变量")
        return np.asarray(mat["RSSI"], dtype=np.float32)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        arr = np.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32)


def score_chart(scores: np.ndarray, threshold: float, title: str = "") -> None:
    """认证分数分布图 — 双栏: 逐窗口分数 + 直方图。"""
    _setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    n = len(scores)
    accept = scores >= threshold
    colors = ["#2E7D32" if a else "#C62828" for a in accept]
    ax1.bar(range(n), scores, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(y=threshold, color="#FF6F00", linestyle="--", linewidth=2,
                label=f"阈值 = {threshold:.4f}")
    ax1.set_xlabel("窗口序号", fontsize=_FS["large"])
    ax1.set_ylabel("认证分数 (0~1, 越高越可信)", fontsize=_FS["large"])
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{title} | 接受率: {np.mean(accept):.1%} "
                  f"(通过数 {int(np.sum(accept))}/{n})", fontsize=_FS["large"])
    ax1.legend(fontsize=_FS["normal"], loc="upper right")
    ax1.grid(axis="y", alpha=0.2)
    ax2.hist(scores[accept], bins=20, alpha=0.6, color="#2E7D32",
             label="接受 (≥阈值)", density=True)
    ax2.hist(scores[~accept], bins=20, alpha=0.6, color="#C62828",
             label="拒绝 (<阈值)", density=True)
    ax2.axvline(x=threshold, color="#FF6F00", linestyle="--", linewidth=2,
                label=f"阈值 = {threshold:.4f}")
    ax2.set_xlabel("认证分数", fontsize=_FS["large"])
    ax2.set_ylabel("概率密度", fontsize=_FS["large"])
    ax2.legend(fontsize=_FS["normal"], loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_model_status(source: str) -> None:
    svm_path = get_model_path(source, "svm")
    cnn_path = get_model_path(source, "cnn")
    cache = st.session_state.pipeline_cache
    cache_count = len(cache._cache_index) if cache else 0
    c1, c2, c3 = st.columns(3)
    items = [(c1, "SVM", svm_path.exists()),
             (c2, "CNN", cnn_path.exists()),
             (c3, "缓存", cache_count > 0)]
    for col, name, ok in items:
        if name == "缓存":
            cls = "model-ready" if ok else "model-missing"
            status = f"✅ {cache_count} 项" if ok else "⚠️ 无"
            col.markdown(f'<div class="{cls}">💾 {name}: {status}</div>',
                         unsafe_allow_html=True)
        else:
            cls = "model-ready" if ok else "model-missing"
            col.markdown(f'<div class="{cls}">{"✅" if ok else "⚠️"} '
                         f'{name}: {"已训练" if ok else "未训练"}</div>',
                         unsafe_allow_html=True)


def show_metrics(sm: dict) -> None:
    if not sm:
        return
    st.markdown("---")
    st.subheader("📊 评估指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HTER ↓", f"{sm['mean_hter']:.4f}")
    c2.metric("FAR ↓", f"{sm['mean_far']:.4f}")
    c3.metric("FRR ↓", f"{sm['mean_frr']:.4f}")
    c4.metric("准确率 ↑", f"{sm['global_accuracy']:.4f}")
    c5.metric("F1 ↑", f"{sm.get('global_f1', 0):.4f}")


def estimate_memory(source: str, **params) -> dict:
    ws = params.get("window_size", 200)
    ss = params.get("step_size", 100)
    mf = params.get("max_files_per_subject") or (220 if source == "csi" else 4)
    n_subjects = 19 if source == "csi" else 5
    est_samples = mf * n_subjects
    est_win_per_sample = (1000 - ws) // ss + 1
    est_win = est_samples * est_win_per_sample
    raw_mb = est_samples * 1000 * 270 * 8 / 1e6 if source == "csi" else 160
    win_mb = est_win * ws * (270 if source == "csi" else 52) * 4 / 1e6
    return {"原始数据": f"{raw_mb:.0f} MB", "窗口数据": f"{win_mb:.0f} MB",
            "优化后预估": f"{win_mb * 0.3:.0f} MB",
            "样本数": est_samples, "预估窗口数": est_win,
            "建议批次": max(16, int(win_mb / 500))}
