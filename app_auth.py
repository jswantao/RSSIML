# -*- coding: utf-8 -*-
"""身份认证交互式系统 — 注册阶段 + 认证阶段。
"""
import gc
import json
import logging
import pickle
import tempfile
import threading
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Callable, Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

from scripts.app_utils import (
    FONT_SIZES as _FS,
    build_windows as _build_windows,
    extract_features_for_auth as _extract_features_for_auth,
    find_processed_file as _find_processed_file,
    save_experiment_subfigures as _save_experiment_subfigures,
    setup_paper_style as _setup_paper_style,
    slice_rssi as _slice_rssi,
)
from scripts.build_sliding_windows import WindowBuilder, WindowConfig
from scripts.config import PipelineConfig
from scripts.pipeline_runner import AuthPipeline, PipelineCache
from scripts.process_features_pca_norm import FeatureExtractor, PreprocessConfig
from scripts.models import (
    CNNInference,
    clear_gpu_memory,
    log_training,
    svm_scores,
)
from scripts.models.memory import get_memory_monitor
from scripts.log_server import WebSocketLogHandler, ensure_server, write_log_html

warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message="resource_tracker")

# ══════════════════════════════════════════════════════════════════════════════
# 基础配置
# ══════════════════════════════════════════════════════════════════════════════
_CONFIG = PipelineConfig.from_root()
_CONFIG.ensure_dirs()
_memory_monitor = get_memory_monitor()

st.set_page_config(
    page_title="身份认证系统", page_icon="🔐", layout="wide",
    initial_sidebar_state="expanded",
)

DataSource = Literal["rssi", "csi"]

# ══════════════════════════════════════════════════════════════════════════════
# 专用日志器
# ══════════════════════════════════════════════════════════════════════════════
app_logger = logging.getLogger("rssi_app")
app_logger.propagate = False
app_logger.setLevel(logging.INFO)
for h in list(app_logger.handlers):
    app_logger.removeHandler(h)

# ══════════════════════════════════════════════════════════════════════════════
# Session State 管理器
# ══════════════════════════════════════════════════════════════════════════════
class SessionStateManager:
    DEFAULTS = {
        "training_active": False,
        "training_cancelled": False,
        "training_launched": False,
        "training_result": None,
        "study_results": None,
        "compare_results": None,
        "exp_results": None,
        "training_error": None,
        "training_progress": 0.0,
        "training_message": "",
        "pipeline_cache": None
    }

    @classmethod
    def init(cls):
        for k, v in cls.DEFAULTS.items():
            if k not in st.session_state:
                st.session_state[k] = v
        if st.session_state.pipeline_cache is None:
            st.session_state.pipeline_cache = PipelineCache(_CONFIG)

    # 训练生命周期
    @classmethod
    def reset_training(cls):
        st.session_state.training_active = True
        st.session_state.training_cancelled = False
        st.session_state.training_launched = False
        st.session_state.training_result = None
        st.session_state.study_results = None
        st.session_state.compare_results = None
        st.session_state.exp_results = None
        st.session_state.training_error = None
        st.session_state.training_progress = 0.0
        st.session_state.training_message = ""

    @classmethod
    def complete_training(cls):
        st.session_state.training_active = False

    @classmethod
    def mark_launched(cls):
        st.session_state.training_launched = True

    @classmethod
    def is_launched(cls):
        return st.session_state.get("training_launched", False)

    @classmethod
    def mark_cancelled(cls):
        st.session_state.training_cancelled = True

    @classmethod
    def is_running(cls):
        return st.session_state.training_active

    @classmethod
    def is_cancelled(cls):
        return st.session_state.training_cancelled

    # 进度 & 结果
    @classmethod
    def update_progress(cls, fraction, message=""):
        st.session_state.training_progress = fraction
        if message:
            st.session_state.training_message = message

    @classmethod
    def set_error(cls, error):
        st.session_state.training_error = error

# ══════════════════════════════════════════════════════════════════════════════
# WebSocket 日志处理器
# ══════════════════════════════════════════════════════════════════════════════
if not any(isinstance(h, WebSocketLogHandler) for h in app_logger.handlers):
    app_logger.addHandler(WebSocketLogHandler())


# ══════════════════════════════════════════════════════════════════════════════
# 异常
# ══════════════════════════════════════════════════════════════════════════════
class TrainingCancelledError(Exception):
    pass

def _check_cancelled():
    if SessionStateManager.is_cancelled():
        raise TrainingCancelledError("训练已被手动终止")


# ══════════════════════════════════════════════════════════════════════════════
# 字体和样式
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_font():
    for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/PingFang.ttc"]:
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return fm.FontProperties()

FONT = _get_font()
if FONT:
    plt.rcParams['font.family'] = FONT.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

st.markdown("""<style>
.auth-header { font-size: 2rem; color: #1E88E5; text-align: center; margin-bottom: 0.5rem; }
.phase-label { font-size: 0.9rem; color: #888; text-align: center; margin-bottom: 1rem; }
.accept-box { padding: 20px; border-radius: 10px; background: #e8f5e9; border: 2px solid #4CAF50; text-align: center; }
.reject-box { padding: 20px; border-radius: 10px; background: #fbe9e7; border: 2px solid #F44336; text-align: center; }
.metric-box { padding: 10px; border-radius: 8px; background: #f0f2f6; text-align: center; margin: 5px 0; }
.model-ready { background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 10px; border-radius: 5px; }
.model-missing { background: #fff3e0; border-left: 4px solid #FF9800; padding: 10px; border-radius: 5px; }
.log-container { max-height: 300px; overflow-y: auto; background: #1e1e1e; color: #d4d4d4;
    padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 0.85rem; line-height: 1.3; }
.log-info { color: #6A9955; } .log-warning { color: #CE9178; } .log-error { color: #F44747; }
.training-status { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.training-running { background: #E3F2FD; border-left: 4px solid #1E88E5; }
.training-cancelled { background: #FFF3E0; border-left: 4px solid #FF9800; }
.training-completed { background: #E8F5E9; border-left: 4px solid #4CAF50; }
.training-error { background: #FFEBEE; border-left: 4px solid #F44336; }
.memory-info { font-size: 0.8rem; color: #666; padding: 5px; background: #f8f9fa; border-radius: 3px; }
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 进度回调工厂
# ══════════════════════════════════════════════════════════════════════════════
def _make_progress_callback(start_fraction=0.15, end_fraction=0.55, op_ui=None):
    span = end_fraction - start_fraction
    def callback(current, total, message=""):
        fraction = current / max(total, 1)
        progress = start_fraction + fraction * span
        SessionStateManager.update_progress(progress, message)
        if op_ui:
            op_ui.update_progress(progress, message)
    return callback


# ══════════════════════════════════════════════════════════════════════════════
# 训练操作 UI 组件（实时日志显示）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainingOperationUI:
    start_key: str
    cancel_key: str
    _status: Any = None
    _progress: Any = None

    def render_controls(self, start_label="🚀 开始训练", on_start=None):
        is_running = SessionStateManager.is_running()
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(start_label, type="primary", key=self.start_key, disabled=is_running):
                if on_start:
                    on_start()
        with col2:
            if is_running:
                if st.button("🛑 终止训练", type="secondary", key=self.cancel_key):
                    SessionStateManager.mark_cancelled()
                    st.rerun()
        self._status = st.empty()
        self._progress = st.empty()

    def update_progress(self, fraction, message=""):
        if self._progress:
            self._progress.progress(fraction, text=message)

    def update_status(self, css_class, message):
        if self._status:
            self._status.markdown(f'<div class="training-status {css_class}"><strong>{message}</strong></div>', unsafe_allow_html=True)

    def render_results(self, result_key="training_result"):
        has_result = st.session_state.get(result_key) is not None
        has_error = st.session_state.training_error is not None
        if has_result:
            self.update_status("training-completed", "✅ 训练完成！")
        elif has_error:
            self.update_status("training-error", f"❌ 训练失败: {st.session_state.training_error}")


# ══════════════════════════════════════════════════════════════════════════════
# 抽象训练执行器（使用 app_logger）
# ══════════════════════════════════════════════════════════════════════════════
class TrainingExecutor(ABC):
    def __init__(self, mtype: str, source: DataSource):
        self.mtype = mtype
        self.source = source
        self.logger = app_logger
        self._op_ui: TrainingOperationUI | None = None

    def run(self):
        try:
            SessionStateManager.mark_launched()
            return self._execute_protected()
        except TrainingCancelledError:
            self._handle_cancelled()
            raise
        except Exception as e:
            self._handle_error(e)
            raise
        finally:
            SessionStateManager.complete_training()
            clear_gpu_memory()
            gc.collect()

    def _execute_protected(self):
        self._on_start()
        _check_cancelled()
        result = self._execute_core()
        self._on_success(result)
        return result

    def _on_start(self):
        self._update_ui(0.05, "初始化流水线...", "training-running", "🔄 训练进行中...")
        self.logger.info(f"开始 {self.mtype} 训练")
        _memory_monitor.log_status()

    def _on_success(self, result):
        self._update_ui(1.0, "训练完成！", "training-completed", "✅ 训练完成！")
        if isinstance(result, dict):
            hter = result.get("system_metrics", {}).get("mean_hter", "N/A")
        elif isinstance(result, list) and result:
            hter = next((r["HTER"] for r in result if r.get("HTER") is not None), "N/A")
        else:
            hter = "N/A"
        self.logger.info(f"训练完成 - HTER: {hter}")

    def _handle_cancelled(self):
        SessionStateManager.mark_cancelled()
        self._update_ui(1.0, "已终止", "training-cancelled", "🛑 训练已被手动终止")
        self.logger.warning("训练已被手动终止")

    def _handle_error(self, error: Exception):
        SessionStateManager.set_error(str(error))
        self._update_ui(1.0, "失败", "training-error", f"❌ 训练失败: {error}")
        self.logger.error(f"训练失败: {error}")

    def _update_ui(self, progress, progress_msg, status_css, status_msg):
        SessionStateManager.update_progress(progress, progress_msg)
        if self._op_ui:
            self._op_ui.update_progress(progress, progress_msg)
            self._op_ui.update_status(status_css, status_msg)

    @abstractmethod
    def _execute_core(self):
        ...

    def get_config(self):
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 基础注册训练执行器
# ══════════════════════════════════════════════════════════════════════════════
class BasicTrainingExecutor(TrainingExecutor):
    def __init__(self, source, mtype, pipeline_kwargs, extra_kwargs):
        super().__init__(mtype, source)
        self.pipeline_kwargs = pipeline_kwargs
        self.extra_kwargs = extra_kwargs

    def get_config(self):
        kw = self.pipeline_kwargs
        c = {
            "data_source": self.source,
            "seed": kw.get("seed"),
            "test_size": kw.get("test_size"),
            "window_size": kw.get("window_size"),
            "step_size": kw.get("step_size"),
            "use_pca": kw.get("use_pca"),
            "threshold_method": kw.get("threshold_method"),
            "max_files_per_subject": kw.get("max_files_per_subject"),
            "use_online_svm": kw.get("use_online_svm", False),
            "feature_groups": list(kw.get("feature_groups", [])),
            "csi_denoise": kw.get("csi_denoise"),
        }
        if kw.get("use_online_svm"):
            c["online_kernel"] = kw.get("online_kernel", "linear")
        c.update(self.extra_kwargs)
        return c

    def _execute_core(self):
        _check_cancelled()
        self._update_ui(0.15, "数据划分中...", "training-running", "🔄 训练进行中...")
        kwargs = dict(self.pipeline_kwargs)
        kwargs["progress_callback"] = _make_progress_callback(0.15, 0.55, self._op_ui)
        pipeline = AuthPipeline(**kwargs)

        _check_cancelled()
        self._update_ui(0.60, f"训练 {self.mtype} 模型中...", "training-running", "🔄 训练进行中...")
        self.logger.info(f"开始 {self.mtype} 模型训练...")

        if self.mtype == "SVM":
            result = pipeline.run_svm()
        else:
            result = pipeline.run_cnn(cancel_fn=_check_cancelled, **self.extra_kwargs)

        _check_cancelled()
        self._update_ui(0.85, "评估模型性能...", "training-running", "🔄 训练进行中...")
        _check_cancelled()
        self._update_ui(0.95, "保存模型...", "training-running", "🔄 训练进行中...")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 参数研究执行器
# ══════════════════════════════════════════════════════════════════════════════
class ParamStudyExecutor(TrainingExecutor):
    """参数研究 — 遍历不同样本数量进行训练对比。

    优化策略:
      1. 全量特征仅准备一次 → memmap 落盘 → OS 按需分页
      2. 一次性超参搜索（RandomizedSearchCV）→ 所有规模复用
      3. 子采样使用索引数组代替数据复制 → 零额外内存
      4. 训练器直接传入内存数组 → 跳过文件 I/O
      5. finally 块确保 memmap 清理
    """

    def __init__(self, source, mtype, study_sizes, seed):
        super().__init__(f"{mtype} 参数研究", source)
        self._model = mtype
        self.study_sizes = sorted(study_sizes)
        self.seed = seed
        self._mmap_paths: list[Path] = []  # 追踪 memmap 文件用于清理

    def get_config(self):
        return {"task": "param_study", "study_sizes": self.study_sizes, "seed": self.seed}

    # ── 子采样工具（返回索引，零复制）─────────────────────────────────────

    @staticmethod
    def _stratified_indices(y, ratio, seed):
        """按用户分层子采样，返回索引数组，避免数据复制。

        Args:
            y: 标签列表或数组。
            ratio: 保留比例 (0~1)。
            seed: 随机种子。

        Returns:
            (selected_indices, y_sub): 子采样索引和对应标签。
        """
        from collections import defaultdict
        rng = np.random.default_rng(seed)
        subj_to_idx = defaultdict(list)
        for idx, s in enumerate(y):
            subj_to_idx[s].append(idx)

        selected = []
        for indices in subj_to_idx.values():
            idx_arr = np.array(indices, dtype=np.intp)
            n_keep = max(1, int(len(idx_arr) * ratio))
            keep = rng.choice(idx_arr, size=n_keep, replace=False)
            selected.extend(keep.tolist())

        selected = np.array(sorted(selected), dtype=np.intp)
        y_arr = np.asarray(y) if not isinstance(y, np.ndarray) else y
        return selected, y_arr[selected]

    # ── 内存映射准备 ─────────────────────────────────────────────────────

    def _prepare_memmap(self, max_files):
        """特征写入 memmap 文件，返回磁盘映射引用和 PCA/Scaler。"""
        pipeline = AuthPipeline(
            data_source=self.source, seed=self.seed, test_size=0.2,
            use_pca=False, max_files_per_subject=max_files,
            use_cache=True, use_model_cache=False,
            save_model=False, save_metrics=False, clean_intermediate=False,
            csi_denoise=_CONFIG.csi_denoise,
        )
        prepared = pipeline._prepare()
        # prepared: (x_tr, y_tr, x_te, y_te, data_file[, fe])
        x_tr = np.asarray(prepared[0], dtype=np.float32)
        y_tr = list(prepared[1]) if not isinstance(prepared[1], np.ndarray) else prepared[1].tolist()
        x_te = np.asarray(prepared[2], dtype=np.float32)
        y_te = list(prepared[3]) if not isinstance(prepared[3], np.ndarray) else prepared[3].tolist()
        tr_shape, te_shape = x_tr.shape, x_te.shape
        # 捕获 PCA/Scaler 供模型保存使用
        fe = pipeline._fe if hasattr(pipeline, '_fe') else None
        pca_model = fe.pca if fe is not None else prepared[5].pca if len(prepared) >= 7 else None
        scaler_model = fe.scaler if fe is not None else (prepared[5].scaler if len(prepared) >= 7 else None)
        if len(prepared) >= 6 and hasattr(prepared[5], 'pca'):
            pca_model = prepared[5].pca
            scaler_model = prepared[5].scaler
        elif fe is not None:
            pca_model = fe.pca
            scaler_model = fe.scaler
        del prepared, pipeline
        gc.collect()

        mmap_dir = _CONFIG.cache_dir / "param_study_mmap"
        mmap_dir.mkdir(parents=True, exist_ok=True)
        # 清理旧文件
        for old in mmap_dir.glob("*.dat"):
            old.unlink(missing_ok=True)

        xt_path = mmap_dir / "x_train.dat"
        xe_path = mmap_dir / "x_test.dat"
        self._mmap_paths = [xt_path, xe_path]

        self.logger.info(f"memmap: train {tr_shape}, test {te_shape}")
        # 写入 memmap
        xt_mmap = np.memmap(xt_path, dtype=np.float32, mode='w+', shape=tr_shape)
        xt_mmap[:] = x_tr[:]; xt_mmap.flush()
        xe_mmap = np.memmap(xe_path, dtype=np.float32, mode='w+', shape=te_shape)
        xe_mmap[:] = x_te[:]; xe_mmap.flush()
        del x_tr, x_te, xt_mmap, xe_mmap
        gc.collect()

        return {
            "xt_path": xt_path, "tr_shape": tr_shape, "y_train": y_tr,
            "xe_path": xe_path, "te_shape": te_shape, "y_test": y_te,
            "pca_model": pca_model, "scaler_model": scaler_model,
        }

    # ── 核心执行逻辑 ─────────────────────────────────────────────────────

    def _execute_core(self):
        results = []
        total = len(self.study_sizes)
        max_files = max(self.study_sizes)

        # 非 SVM 模型走独立流水线
        if self._model != "SVM":
            for i, n_files in enumerate(self.study_sizes):
                _check_cancelled()
                results.append(self._run_single_study(n_files, i, total))
            st.session_state.study_results = results
            return results

        mm = None
        try:
            # ── 阶段 1: memmap 特征准备（仅一次）──
            self._update_ui(0.05, f"提取特征 → memmap (max={max_files})...",
                            "training-running", "🔄 参数研究: memmap 准备...")
            self.logger.info(f"参数研究: memmap 特征准备 (max_files={max_files})...")
            mm = self._prepare_memmap(max_files)
            self.logger.info(f"memmap 就绪: train {mm['tr_shape']}, test {mm['te_shape']}")

            # 打开只读 memmap（OS 按需分页，不驻留内存）
            x_tr_full = np.memmap(mm["xt_path"], dtype=np.float32, mode='r', shape=mm["tr_shape"])
            x_te_full = np.memmap(mm["xe_path"], dtype=np.float32, mode='r', shape=mm["te_shape"])
            # 关键修复：将 Python 列表转为 NumPy 数组，确保切片和索引操作正确
            y_tr_full = np.asarray(mm["y_train"])
            y_te_full = np.asarray(mm["y_test"])

            # ── 阶段 2: 在线 SVM 参数 (无需搜索) ──
            self._update_ui(0.10, "准备在线 SVM 训练...", "training-running", "🔄 参数研究: 在线SVM准备...")
            self.logger.info("参数研究: 在线 SVM (SGD+RBFSampler), 固定参数跳过搜索")
            best_params = self._find_best_params(x_tr_full, y_tr_full)
            self.logger.info(f"在线 SVM 参数: {best_params}")

            # ── 阶段 3: 各规模训练 ──
            rank = 1
            for n_files in sorted(self.study_sizes, reverse=True):
                _check_cancelled()
                progress = 0.12 + rank / max(total, 1) * 0.83
                self._update_ui(progress, f"训练: {n_files} files ({rank}/{total})",
                                "training-running", f"🔄 研究: {rank}/{total}")
                self.logger.info(f"参数研究 [{rank}/{total}]: {n_files} files/subject")

                ratio = n_files / max_files
                if ratio >= 1.0:
                    indices = np.arange(len(y_tr_full), dtype=np.intp)
                    y_tr_sub = y_tr_full
                else:
                    indices, y_tr_sub = self._stratified_indices(
                        y_tr_full, ratio, seed=self.seed + n_files)

                entry = self._train_with_data(
                    x_tr_full, indices, y_tr_sub, x_te_full, y_te_full,
                    best_params, n_files, rank, total,
                    pca_model=mm.get("pca_model"), scaler_model=mm.get("scaler_model"))
                results.append(entry)
                st.session_state.study_results = list(results)
                clear_gpu_memory()
                gc.collect()
                rank += 1

        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"参数研究失败: {e}")
            traceback.print_exc()
            # 回退到独立流水线
            for i, n_files in enumerate(self.study_sizes):
                _check_cancelled()
                results.append(self._run_single_study(n_files, i, total))
        finally:
            # 清理 memmap 文件
            for p in self._mmap_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            self._mmap_paths.clear()

        results.reverse()
        self.study_sizes.sort()
        st.session_state.study_results = results
        return results


    @staticmethod
    def _find_best_params(x_tr, y_tr):
        """在线 SVM 固定参数, 无需超参搜索。

        SGDClassifier(hinge) + RBFSampler 核近似:
          - alpha=1e-4: L2 正则化, 默认值在大多数场景表现良好
          - gamma='scale': 自适应特征维度, 避免高维欠拟合
          - n_components=200: RBF 核近似 Fourier 特征数
        """
        return {"alpha": 1e-4, "gamma": "scale", "n_components": 200}


    # ── 单规模训练（索引零复制 + 内存直传）─────────────────────────────

    def _train_with_data(self, x_tr_full, indices, y_tr_sub, x_te, y_te,
                         best_params, n_files, idx, total,
                         pca_model=None, scaler_model=None):
        """在线 SVM 训练 — SGD+RBFSampler, 大幅快于批处理 SVC。

        Args:
            x_tr_full: 全量训练特征 memmap。
            indices: 子采样索引数组。
            y_tr_sub: 子采样后的标签。
            x_te, y_te: 测试集。
            best_params: 在线 SVM 参数 dict (alpha, gamma, n_components)。
            n_files: 当前规模（仅用于日志）。
            idx, total: 进度序号。
            pca_model, scaler_model: 训练时拟合的 PCA/Scaler, 传递给模型供推理使用。
        """
        try:
            from scripts.models import SVMConfig, SVMAuthenticationTrainer
            sc = SVMConfig(threshold_method="youden", random_seed=self.seed, cv_folds=3)
            trainer = SVMAuthenticationTrainer(sc, use_online=True, online_kernel="rbf")

            result = self._train_svm_memory(
                trainer, x_tr_full, indices, y_tr_sub, x_te, y_te,
                pca_model=pca_model, scaler_model=scaler_model,
                data_source=self.source)

            sm = result.get("system_metrics", {})
            entry = {
                "每用户文件数": n_files,
                "HTER": sm.get("mean_hter", 0),
                "FAR": sm.get("mean_far", 0),
                "FRR": sm.get("mean_frr", 0),
                "准确率": sm.get("global_accuracy", 0),
                "HTER_std": sm.get("std_hter", 0),
                "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0),
            }
            self.logger.info(
                f"✓ [{idx}/{total}]: {n_files} files → HTER={sm.get('mean_hter', 'N/A'):.4f}")
            return entry
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"✗ [{idx}/{total}] ({n_files}): {e}")
            return {
                "每用户文件数": n_files,
                "HTER": None, "FAR": None, "FRR": None, "准确率": None,
                "HTER_std": None, "FAR_std": None, "FRR_std": None,
                "错误": str(e),
            }

    @staticmethod
    def _train_svm_memory(trainer, x_full, indices, y_tr, x_te, y_te,
                          pca_model=None, scaler_model=None, data_source=None):
        """索引访问 memmap 后直接传入数组训练, 跳过文件序列化往返。"""
        x_tr_sub = np.asarray(x_full[indices], dtype=np.float32)
        return trainer.train_from_arrays(
            x_tr_sub, y_tr, x_te, y_te,
            pca_model=pca_model, scaler_model=scaler_model,
            data_source=data_source)

    # ── 回退: 独立流水线 ──────────────────────────────────────────────────

    def _run_single_study(self, n_files, index, total):
        try:
            pipeline = AuthPipeline(
                data_source=self.source, seed=self.seed, test_size=0.2,
                use_pca=False, max_files_per_subject=n_files,
                use_cache=True, use_model_cache=False,
                save_model=False, save_metrics=False, clean_intermediate=False,
                csi_denoise=_CONFIG.csi_denoise,
            )
            if self._model == "SVM":
                result = pipeline.run_svm()
            else:
                result = pipeline.run_cnn(epochs=10, batch_size=64, cancel_fn=_check_cancelled)
            sm = result.get("system_metrics", {})
            entry = {
                "每用户文件数": n_files,
                "HTER": sm.get("mean_hter", 0),
                "FAR": sm.get("mean_far", 0),
                "FRR": sm.get("mean_frr", 0),
                "准确率": sm.get("global_accuracy", 0),
                "HTER_std": sm.get("std_hter", 0),
                "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0),
            }
            self.logger.info(
                f"✓ [{index+1}/{total}]: {n_files} files → HTER={sm.get('mean_hter', 'N/A'):.4f}")
            return entry
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"✗ [{index+1}/{total}] ({n_files}): {e}")
            return {
                "每用户文件数": n_files,
                "HTER": None, "FAR": None, "FRR": None, "准确率": None,
                "HTER_std": None, "FAR_std": None, "FRR_std": None, "错误": str(e),
            }

# ══════════════════════════════════════════════════════════════════════════════
# 模型对比执行器
# ══════════════════════════════════════════════════════════════════════════════
class ModelCompareExecutor(TrainingExecutor):
    def __init__(self, source, n_files, seed, test_size):
        super().__init__("模型对比", source)
        self.n_files = n_files
        self.seed = seed
        self.test_size = test_size

    def get_config(self):
        return {"task": "model_compare", "n_files": self.n_files, "seed": self.seed, "test_size": self.test_size}

    def _execute_core(self):
        results = {}
        _check_cancelled()
        self._update_ui(0.1, "训练 SVM (1/2)...", "training-running", "🔄 模型对比: SVM (1/2)...")
        self.logger.info("SVM 训练开始...")
        results["SVM"] = self._train_svm()
        clear_gpu_memory(); gc.collect()

        _check_cancelled()
        self._update_ui(0.5, "训练 CNN (2/2)...", "training-running", "🔄 模型对比: CNN (2/2)...")
        self.logger.info("CNN 训练开始...")
        results["CNN"] = self._train_cnn()
        st.session_state.compare_results = results
        return results

    def _make_pipeline(self, **overrides):
        kw = dict(data_source=self.source, seed=self.seed, test_size=self.test_size,
                  use_pca=False, max_files_per_subject=self.n_files,
                  use_cache=True, use_model_cache=False,
                  save_model=False, save_metrics=False, clean_intermediate=False,
                  csi_denoise=_CONFIG.csi_denoise)
        kw.update(overrides)
        return AuthPipeline(**kw)

    def _train_svm(self):
        try:
            if self.source == "csi":
                result = self._make_pipeline(
                    use_online_svm=True, online_kernel="rbf").run_svm()
            else:
                result = self._make_pipeline().run_svm()
            sm = result.get("system_metrics", {})
            self.logger.info(f"SVM 完成 - HTER: {sm.get('mean_hter', 'N/A')}")
            return {
                "HTER": sm.get("mean_hter", 0), "FAR": sm.get("mean_far", 0),
                "FRR": sm.get("mean_frr", 0), "准确率": sm.get("global_accuracy", 0),
                "HTER_std": sm.get("std_hter", 0), "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0),
            }
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"SVM 训练失败: {e}")
            return {"错误": str(e)}

    def _train_cnn(self):
        try:
            result = self._make_pipeline().run_cnn(epochs=10, batch_size=64, cancel_fn=_check_cancelled)
            sm = result.get("system_metrics", {})
            self.logger.info(f"CNN 完成 - HTER: {sm.get('mean_hter', 'N/A')}")
            return {
                "HTER": sm.get("mean_hter", 0), "FAR": sm.get("mean_far", 0),
                "FRR": sm.get("mean_frr", 0), "准确率": sm.get("global_accuracy", 0),
                "HTER_std": sm.get("std_hter", 0), "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0),
            }
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"CNN 训练失败: {e}")
            return {"错误": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数（加载、认证等）
# ══════════════════════════════════════════════════════════════════════════════

def _get_auth_model_path(source, model_type):
    suffix = "_npy" if source == "csi" else ""
    if model_type == "svm":
        return _CONFIG.model_dir / f"svm_authentication{suffix}.pkl"
    return _CONFIG.model_dir / f"cnn_authentication{suffix}.pt"

@st.cache_resource(show_spinner=False)
def _load_auth_model(source):
    mp = _get_auth_model_path(source, "svm")
    if not mp.exists():
        return None, None, None, None
    try:
        with mp.open("rb") as f:
            model = pickle.load(f)
        # 统一键映射: 旧模型原始 ID → UI 编号 (幂等)
        _normalize_model_keys(model, source)

        pca = getattr(model, "pca_model", None)
        scaler = getattr(model, "scaler_model", None)
        feature_config = getattr(model, "feature_config", None)
        model_src = getattr(model, "data_source", None)

        # 数据源校验
        if model_src and model_src != source:
            app_logger.warning(
                f"数据源不匹配: 模型训练于 '{model_src}', 当前选择 '{source}'。"
                f"特征维度可能不一致。")

        # 日志: Scaler 期望维度
        if scaler is not None:
            sc_n = getattr(scaler, 'n_features_in_', '?')
            app_logger.info(
                f"模型 Scaler 期望 {sc_n} 维, PCA={'已加载' if pca else '无'}, "
                f"feature_config={'已加载' if feature_config else '无'}, "
                f"data_source={model_src or '?'}")

        # 仅当 Scaler 缺失时才回退到处理文件 (PCA 可合法为 None)
        if scaler is None:
            pp = _find_processed_file(source)
            if pp and pp.exists():
                with pp.open("rb") as f:
                    proc = pickle.load(f)
                pca = proc.get("pca_model") or proc.get("pca")
                scaler = proc.get("scaler_model") or proc.get("scaler")
                # 旧模型: 从处理文件 meta 推断特征配置
                if feature_config is None and "meta" in proc:
                    meta = proc["meta"]
                    feature_config = {
                        "feature_groups": meta.get("feature_groups", ["spectral", "statistical"]),
                        "low_freq_bins": meta.get("low_freq_bins", 16),
                        "denoise": meta.get("denoise"),
                        "denoise_kernel": meta.get("denoise_kernel", 5),
                    }
        return model, pca, scaler, feature_config
    except Exception as e:
        app_logger.error(f"模型加载失败: {e}")
        return None, None, None, None

@st.cache_resource(show_spinner=False)
def _load_cnn_model(source):
    mp = _get_auth_model_path(source, "cnn")
    if not mp.exists():
        return None
    try:
        cnn = CNNInference(mp)
        if not cnn.is_authentication:
            return None
        # 统一键映射: 旧模型原始 ID → UI 编号 (幂等)
        _normalize_model_keys(cnn, source)
        return cnn
    except Exception as e:
        app_logger.error(f"CNN加载失败: {e}")
        return None

def _normalize_model_keys(model, source):
    """统一规范化模型中的用户键为 UI 编号, 幂等操作。

    处理 verifiers, thresholds, subjects, label_encoder 四个属性。
    旧模型存原始 ID (12-30/FXY), 新模型已映射 (1-19/1-5), 均转为 UI 编号。
    """
    fwd = _CONFIG.subject_map(source)
    if not fwd:
        return  # 无映射表, 跳过

    def _mapped(k):
        return fwd.get(str(k), str(k))

    if hasattr(model, 'verifiers'):
        model.verifiers = {_mapped(k): v for k, v in model.verifiers.items()}
    if hasattr(model, 'thresholds'):
        model.thresholds = {_mapped(k): v for k, v in model.thresholds.items()}
    if hasattr(model, 'subjects'):
        model.subjects = [_mapped(s) for s in model.subjects]
    if hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'classes_'):
        model.label_encoder.classes_ = np.array(
            [_mapped(c) for c in model.label_encoder.classes_])
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'classes_'):
        model.encoder.classes_ = np.array(
            [_mapped(c) for c in model.encoder.classes_])


def _get_available_subjects(model, cnn_model, source):
    """返回当前可用用户列表, 优先使用已加载模型的 subjects。"""
    avail = (_CONFIG.DEFAULT_SUBJECTS.get(source, []) or [])
    if cnn_model and hasattr(cnn_model, 'subjects') and cnn_model.subjects:
        avail = list(cnn_model.subjects)
    if model and hasattr(model, 'subjects') and model.subjects:
        avail = list(model.subjects)
    return avail




def _combine_csi_files(data_dir: Path, subject: str, trial: int = 0,
                       max_files: int | None = None) -> np.ndarray | None:
    """将同一用户多个动作的 CSI 文件沿时间轴拼接为连续样本。

    CSI 单文件仅 ~5s (1000 时间步), 拼接所有动作后得到连续长样本,
    与 RSSI 200s 样本在规模上可比较。

    Args:
        data_dir: CSI 数据目录 (WiFi/)。
        subject: 用户 ID。
        trial: trial 编号 (0-19), 0 表示使用每个动作的第一个可用 trial。
        max_files: 最大拼接文件数, None 表示不限制。

    Returns:
        (total_time_steps, n_features) float32 拼接矩阵, 或 None。
    """
    import re
    from scripts.data_loader import load_npy_matrix
    disk_subj = _CONFIG.subject_unmap("csi").get(subject, subject)
    pattern = re.compile(rf"^{disk_subj}_(\d+)_(\d+)\.npy$")
    # 收集该用户的所有文件, 按 action 分组
    by_action: dict[int, list[Path]] = {}
    for fp in data_dir.glob(f"{disk_subj}_*.npy"):
        m = pattern.match(fp.name)
        if not m:
            continue
        action = int(m.group(1))
        t = int(m.group(2))
        by_action.setdefault(action, []).append((t, fp))
    if not by_action:
        return None
    # 每个 action 选一个 trial
    selected = []
    for action in sorted(by_action):
        files = sorted(by_action[action], key=lambda x: x[0])
        if trial > 0 and trial <= len(files):
            selected.append(files[trial - 1][1])
        elif files:
            selected.append(files[0][1])  # 第一个可用 trial
    if max_files:
        selected = selected[:max_files]
    if not selected:
        return None
    # 加载并拼接
    arrays = []
    for fp in selected:
        try:
            arr = load_npy_matrix(fp)
            arrays.append(arr)
        except (ValueError, OSError):
            continue
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0).astype(np.float32)



def _load_upload(uploaded, source):
    try:
        if source == "rssi":
            mat = loadmat(uploaded)
            if "RSSI" not in mat:
                raise ValueError("缺少 'RSSI' 键")
            data = np.asarray(mat["RSSI"], dtype=np.float32)
        else:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as t:
                t.write(uploaded.read())
                tp = Path(t.name)
            try:
                data = np.load(tp)
            finally:
                tp.unlink(missing_ok=True)
            if data.ndim != 2:
                raise ValueError(f"维度异常: {data.shape}")
            data = np.asarray(data.T, dtype=np.float32)
        if data.size == 0:
            raise ValueError("文件为空")
        return data
    except Exception as e:
        raise ValueError(f"文件加载失败: {e}")

def _estimate_memory_usage(source, **params):
    ws = params.get("window_size", 200)
    ss = params.get("step_size", 100)
    n_files = params.get("max_files_per_subject", 220)
    if source == "rssi":
        n_ch, n_time, n_samples = 52, 20000, 20
    else:
        n_ch, n_time = 270, 1000  # 270 subcarriers (80MHz 802.11ac)
        n_samples = n_files if n_files else 1100
    raw_mb = (n_samples * n_time * n_ch * 4) / (1024 * 1024)
    n_windows = WindowBuilder.estimate_num_windows(n_time, ws, ss)
    window_mb = (n_samples * n_windows * n_ch * ws * 4) / (1024 * 1024)
    feature_mb = window_mb * 0.1
    model_mb = max(10, window_mb * 0.01)
    optimized_mb = window_mb * 0.3 if source == "csi" else window_mb * 0.5
    return {
        "原始数据": f"{raw_mb:.0f} MB",
        "窗口数据": f"{window_mb:.0f} MB",
        "优化后预估": f"{optimized_mb + feature_mb + model_mb:.0f} MB",
        "样本数": n_samples,
        "预估窗口数": int(n_samples * n_windows),
        "建议批次": max(8, int(4096 / max(1, optimized_mb)) * 8),
    }

def _show_metrics(sm):
    """论文级系统指标展示 — HTER/FAR/FRR + 每用户详情。"""
    if not sm:
        st.info("暂无指标数据")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    mh = sm.get("mean_hter", 0)
    c1.metric("HTER ↓", f"{mh:.4f}" if isinstance(mh, float) else str(mh),
              help="Half Total Error Rate = (FAR+FRR)/2")
    c2.metric("FAR ↓", f'{sm.get("mean_far", 0):.4f}',
              help="False Acceptance Rate")
    c3.metric("FRR ↓", f'{sm.get("mean_frr", 0):.4f}',
              help="False Rejection Rate")
    c4.metric("Accuracy ↑", f'{sm.get("global_accuracy", 0):.4f}',
              help="全局准确率")
    c5.metric("F1 ↑", f'{sm.get("global_f1", 0):.4f}',
              help="F1 Score")
    if um := sm.get("user_metrics"):
        with st.expander("📊 各用户详细指标", expanded=False):
            df = pd.DataFrame([
                {"用户": u, "FAR": f'{m["far"]:.4f}', "FRR": f'{m["frr"]:.4f}',
                 "HTER": f'{m["hter"]:.4f}', "阈值": f'{m.get("threshold", 0):.4f}',
                 "Genuine": m.get("n_genuine_tests", 0),
                 "Impostor": m.get("n_impostor_tests", 0)}
                for u, m in um.items()
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
            # 每用户 HTER 柱状图
            _setup_paper_style()
            fig, ax = plt.subplots(figsize=(8, 3.5))
            users = list(um.keys())
            hters = [um[u]["hter"] for u in users]
            ax.bar(range(len(users)), hters, color='#4472C4', alpha=0.85, edgecolor='white')
            ax.axhline(y=mh, color='#C00000', linestyle='--', linewidth=1.5,
                       label=f'平均 HTER = {mh:.4f}')
            ax.set_xticks(range(len(users)))
            ax.set_xticklabels(users, fontsize=_FS["normal"])
            ax.set_ylabel("HTER (半总错误率)", fontsize=_FS["large"])
            ax.set_xlabel("用户", fontsize=_FS["large"])
            ax.set_ylim(0, min(1.0, max(hters) * 1.3))
            ax.legend(fontsize=_FS["normal"], loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)


def _score_chart(scores, threshold, title=""):
    """论文级认证分数分布图 — 中文标注, 含指标说明。"""
    _setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5),
                                    gridspec_kw={'width_ratios': [3, 1]})
    n = len(scores)
    accept = scores >= threshold
    # 左: 逐窗口分数
    colors = ['#2E7D32' if a else '#C62828' for a in accept]
    ax1.bar(range(n), scores, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(y=threshold, color='#FF6F00', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold:.4f}')
    ax1.set_xlabel("窗口序号", fontsize=_FS["large"])
    ax1.set_ylabel("认证分数 (0~1, 越高越可信)", fontsize=_FS["large"])
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{title} | 接受率: {np.mean(accept):.1%} "
                  f"(通过数 {int(np.sum(accept))}/{n})", fontsize=_FS["large"])
    ax1.legend(fontsize=_FS["normal"], loc='upper right')
    ax1.grid(axis='y', alpha=0.2)
    # 右: 分数分布直方图
    ax2.hist(scores[accept], bins=20, alpha=0.6, color='#2E7D32', label='接受 (≥阈值)', density=True)
    ax2.hist(scores[~accept], bins=20, alpha=0.6, color='#C62828', label='拒绝 (<阈值)', density=True)
    ax2.axvline(x=threshold, color='#FF6F00', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold:.4f}')
    ax2.set_xlabel("认证分数", fontsize=_FS["large"])
    ax2.set_ylabel("概率密度", fontsize=_FS["large"])
    ax2.legend(fontsize=_FS["normal"], loc='upper right')
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

def _show_model_status(source):
    svm_path = _get_auth_model_path(source, "svm")
    cnn_path = _get_auth_model_path(source, "cnn")
    svm_ok = svm_path.exists()
    cnn_ok = cnn_path.exists()
    cache = st.session_state.pipeline_cache
    cache_count = len(cache._cache_index) if cache else 0
    c1, c2, c3 = st.columns(3)
    for col, name, ok in [(c1, "SVM", svm_ok), (c2, "CNN", cnn_ok), (c3, "缓存", cache_count > 0)]:
        if name == "缓存":
            cls = "model-ready" if ok else "model-missing"
            status = f"✅ {cache_count} 项" if ok else "⚠️ 无"
            col.markdown(f'<div class="{cls}">💾 {name}: {status}</div>', unsafe_allow_html=True)
        else:
            cls = "model-ready" if ok else "model-missing"
            col.markdown(f'<div class="{cls}">{"✅" if ok else "⚠️"} {name}: {"已训练" if ok else "未训练"}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 训练启动（防重入）
# ══════════════════════════════════════════════════════════════════════════════
_training_lock = threading.Lock()

def _launch_training(executor_factory, result_key, op_ui=None):
    if st.session_state.get(result_key) is not None:
        return
    if not _training_lock.acquire(blocking=False):
        app_logger.warning("训练已在执行中，忽略重复启动")
        return
    try:
        t_start = time.time()
        executor = None
        try:
            executor = executor_factory()
            executor._op_ui = op_ui
            result = executor.run()
            if result is not None:
                st.session_state[result_key] = result
            if isinstance(result, dict):
                if "system_metrics" in result:
                    metrics = result["system_metrics"]
                elif any(k.startswith("E") and k[1:].isdigit() for k in result):
                    # 实验嵌套结果 → 提取最佳 HTER
                    best_hter = 1.0
                    for exp_data in result.values():
                        if isinstance(exp_data, dict):
                            for v in exp_data.values():
                                if isinstance(v, dict):
                                    h = v.get("mean_hter", v.get("HTER"))
                                    if h is not None:
                                        best_hter = min(best_hter, float(h))
                        elif isinstance(exp_data, list):
                            for r in exp_data:
                                h = r.get("HTER")
                                if h is not None:
                                    best_hter = min(best_hter, float(h))
                    metrics = {"mean_hter": best_hter if best_hter < 1.0 else "N/A"}
                else:
                    metrics = {}
            elif isinstance(result, list) and result:
                best = min((r for r in result if r.get("HTER") is not None),
                          key=lambda r: r["HTER"], default={})
                metrics = {"mean_hter": best.get("HTER", "N/A")}
            else:
                metrics = {}
            log_training(
                pipeline_config=_CONFIG,
                model_type=executor.mtype, task_type="authentication",
                data_source=executor.source, status="success",
                duration=time.time() - t_start,
                config=executor.get_config(),
                metrics=metrics,
            )
        except TrainingCancelledError:
            if executor:
                log_training(
                    pipeline_config=_CONFIG,
                    model_type=executor.mtype, task_type="authentication",
                    data_source=executor.source, status="cancelled",
                    duration=time.time() - t_start,
                    config=executor.get_config(),
                )
        except Exception as e:
            err_msg = str(e)
            mtype = executor.mtype if executor else "unknown"
            source = executor.source if executor else "rssi"
            cfg = executor.get_config() if executor else {}
            log_training(
                pipeline_config=_CONFIG,
                model_type=mtype, task_type="authentication",
                data_source=source, status="failed",
                duration=time.time() - t_start,
                config=cfg, error=err_msg[:500],
            )
            st.session_state.training_error = err_msg
            app_logger.error(f"训练异常:\n{traceback.format_exc()}")
    finally:
        _training_lock.release()


# ══════════════════════════════════════════════════════════════════════════════
# UI 注册阶段（三个 Tab）
# ══════════════════════════════════════════════════════════════════════════════
def _render_register_basic(source):
    st.subheader("🔧 基础注册训练")
    st.caption(f"数据源: {'静态 (RSSI/MAT)' if source == 'rssi' else '动态 (CSI/NPY)'}")
    _show_model_status(source)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.number_input("随机种子", 0, 999, 42, key="reg_seed")
        test_size = st.slider("测试比例", 0.1, 0.4, 0.2, 0.05, key="reg_ts")
    with c2:
        ws = st.number_input("窗口大小", 50, 500, 200, 50, key="reg_ws")
        ss = st.number_input("步长", 10, 200, 100, 10, key="reg_ss")
    with c3:
        threshold_method = st.selectbox("阈值方法", ["youden", "quantile", "fixed", "eer"], key="reg_tm")
        max_files = None
        if source == "csi":
            na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
            max_files = st.number_input("每用户文件数", na, na * 20, na * 8, na, key="reg_mf")

    mtype = st.radio("模型类型", ["SVM", "CNN"], horizontal=True, key="reg_mt")

    with st.expander("⚙️ 高级选项", expanded=False):
        use_pca = st.checkbox("PCA 降维", False, key="reg_pca")
        feature_groups = st.multiselect(
            "特征组",
            ["spectral", "statistical", "temporal"],
            default=["spectral", "statistical", "temporal"],
            key="reg_fg",
            help="spectral: FFT 频域特征 | statistical: 统计特征 | temporal: 时域特征")
        use_cache = st.checkbox("启用缓存", True, key="reg_cache")
        clean_intermediate = st.checkbox("清理中间文件", True, key="reg_clean")
        csi_denoise = None
        if source == "csi":
            csi_denoise = st.selectbox(
                "CSI 降噪方法", ["无", "hampel", "savgol", "butterworth"],
                key="reg_denoise",
                help="hampel: 脉冲异常值去除 | "
                     "savgol: 多项式平滑 (保留边缘) | "
                     "butterworth: 巴特沃斯低通 (20Hz截止)")
            csi_denoise = None if csi_denoise == "无" else csi_denoise

        use_online_svm = False
        online_kernel = "linear"
        if mtype == "SVM" and source == "csi":
            use_online_svm = st.checkbox(
                "在线SVM (SGD, 增量学习)",
                False, key="reg_online",
                help="使用 SGDClassifier 替代 RBF SVC, 支持增量训练。"
                     "适合 CSI 大规模窗口数据。")
            if use_online_svm:
                online_kernel = st.selectbox(
                    "在线SVM核函数",
                    ["linear", "rbf"], key="reg_online_kernel",
                    help="linear: 线性 SGD, 最快 | "
                         "rbf: RBFSampler 核近似 + SGD, 逼近批处理 RBF SVC 效果")

    extra = {}
    if mtype == "CNN":
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            extra["epochs"] = st.number_input("Epochs", 5, 100, 20, 5, key="reg_ep")
        with ec2:
            extra["batch_size"] = st.number_input("Batch", 8, 256, 64, 8, key="reg_bs")
        with ec3:
            extra["learning_rate"] = st.number_input("LR", 1e-5, 0.1, 1e-3, format="%.5f", key="reg_lr")

    kw = dict(
        data_source=source, seed=seed, test_size=test_size, window_size=ws,
        step_size=ss, use_pca=use_pca, threshold_method=threshold_method,
        use_cache=use_cache, clean_intermediate=clean_intermediate,
        max_files_per_subject=max_files, use_online_svm=use_online_svm,
        online_kernel=online_kernel,
        feature_groups=tuple(feature_groups),
        csi_denoise=csi_denoise,
    )
    mem_info = _estimate_memory_usage(source, **kw)
    with st.expander("💾 内存评估", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("原始数据", mem_info["原始数据"])
        mc2.metric("窗口数据", mem_info["窗口数据"])
        mc3.metric("优化后", mem_info["优化后预估"])
        st.caption(f"样本数: {mem_info['样本数']} | 窗口数: {mem_info['预估窗口数']:,} | 建议批次: {mem_info['建议批次']}")

    op_ui = TrainingOperationUI(start_key="reg_start", cancel_key="reg_cancel")
    op_ui.render_controls(start_label="🚀 开始注册训练", on_start=SessionStateManager.reset_training)

    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("training_result") is not None
    has_error = st.session_state.training_error is not None

    if is_running or has_result or has_error:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(
                lambda: BasicTrainingExecutor(source, mtype, dict(kw), extra),
                "training_result", op_ui
            )
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("training_result") is not None
            has_error = st.session_state.training_error is not None
        if has_result or has_error:
            op_ui.render_results()
        if has_result:
            _show_metrics(st.session_state.training_result.get("system_metrics", {}))


def _render_param_study(source):
    st.subheader("📈 注册参数研究")
    st.caption("研究训练样本数对认证性能的影响")
    study_config = _CONFIG.PARAM_STUDY_DEFAULTS[source]
    study_sizes = st.multiselect("研究样本数", study_config["sizes"], default=study_config["default"], key="study_sizes")
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input("随机种子", 0, 999, 42, key="study_seed")
    with c2:
        mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="study_mt")

    op_ui = TrainingOperationUI(start_key="study_start", cancel_key="study_cancel")
    def on_start():
        if not study_sizes:
            st.warning("请至少选择一个样本数")
            st.stop()
        SessionStateManager.reset_training()
    op_ui.render_controls(start_label="▶️ 开始参数研究", on_start=on_start)

    is_running = SessionStateManager.is_running()
    has_result = st.session_state.study_results is not None

    if is_running or has_result:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(
                lambda: ParamStudyExecutor(source, mtype, study_sizes, seed),
                "study_results", op_ui
            )
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.study_results is not None
        if has_result or st.session_state.training_error:
            op_ui.render_results("study_results")
        if has_result:
            _render_study_results(st.session_state.study_results, mtype)

def _render_study_results(results, mtype):
    """论文级参数研究结果 — 双栏表格 + 误差带折线图。"""
    if not isinstance(results, list) or not results:
        st.info("暂无参数研究结果。")
        return

    df = pd.DataFrame(results)
    valid = df[df["HTER"].notna()].copy()

    st.markdown("---")
    st.subheader("📊 参数研究结果")

    # ── 表格 ────────────────────────────────────────────────────────────────
    has_std = "HTER_std" in df.columns and df["HTER_std"].notna().any()
    if has_std:
        tbl_cols = {
            "每用户文件数": "训练样本 / 用户",
            "HTER": "HTER", "FAR": "FAR", "FRR": "FRR",
            "准确率": "准确率",
            "HTER_std": "HTER (std)", "FAR_std": "FAR (std)", "FRR_std": "FRR (std)",
        }
    else:
        tbl_cols = {
            "每用户文件数": "训练样本 / 用户",
            "HTER": "HTER", "FAR": "FAR", "FRR": "FRR",
            "准确率": "准确率",
        }
    df_tbl = df.rename(columns=tbl_cols)
    avail_cols = [c for c in tbl_cols.values() if c in df_tbl.columns]
    df_tbl = df_tbl[avail_cols]

    fmt = {c: "{:.4f}" for c in avail_cols if "样本" not in c}
    st.dataframe(
        df_tbl.style.format(fmt, na_rep="—"),
        hide_index=True, use_container_width=True,
    )

    if len(valid) < 2:
        return

    # ── 图表 ────────────────────────────────────────────────────────────────
    _setup_paper_style()
    x = valid["每用户文件数"].values.astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- 左: 错误率曲线 ---
    colors = {"HTER": "#C00000", "FAR": "#4472C4", "FRR": "#ED7D31"}
    markers = {"HTER": "o", "FAR": "s", "FRR": "^"}
    for metric, color in colors.items():
        y = valid[metric].values.astype(float)
        ax1.plot(x, y, marker=markers[metric], color=color, lw=2, ms=7,
                 label=metric, zorder=3)
        # 标准差阴影
        std_col = f"{metric}_std"
        if has_std and std_col in valid.columns:
            std = valid[std_col].values.astype(float)
            if not np.all(std == 0):
                ax1.fill_between(x, np.maximum(0, y - std), y + std,
                                 color=color, alpha=0.12)
    ax1.set_xlabel("训练样本 / 用户", fontsize=_FS["large"])
    ax1.set_ylabel("错误率 (HTER/FAR/FRR)", fontsize=_FS["large"])
    ax1.set_title("(a) 错误率随样本数变化", fontsize=_FS["large"], fontweight='medium')
    ax1.legend(fontsize=_FS["normal"], framealpha=0.85, loc='upper right',
               title="HTER = (FAR+FRR)/2")
    ax1.grid(True, alpha=0.2, lw=0.5)
    ax1.set_ylim(0, min(1.0, valid[["HTER", "FAR", "FRR"]].max().max() * 1.35))
    if len(x) <= 6:
        ax1.set_xticks(x)

    # --- 右: 准确率曲线 ---
    acc = valid["准确率"].values.astype(float)
    ax2.plot(x, acc, 'D-', color='#2E7D32', lw=2, ms=8, label='准确率', zorder=3)
    ax2.set_xlabel("训练样本 / 用户", fontsize=_FS["large"])
    ax2.set_ylabel("准确率", fontsize=_FS["large"])
    ax2.set_title("(b) 准确率随样本数变化", fontsize=_FS["large"], fontweight='medium')
    ax2.grid(True, alpha=0.2, lw=0.5)
    y_bot = max(0.0, acc.min() - 0.08)
    ax2.set_ylim(y_bot, min(1.0, acc.max() + 0.05))
    if len(x) <= 6:
        ax2.set_xticks(x)

    fig.suptitle(f"参数研究 — {mtype}", fontsize=_FS["title"], fontweight='bold', y=1.01)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # ── 最佳样本数总结 ──────────────────────────────────────────────────────
    best_idx = valid["HTER"].values.argmin()
    best_n = int(valid["每用户文件数"].iloc[best_idx])
    best_hter = float(valid["HTER"].iloc[best_idx])
    col1, col2, col3 = st.columns(3)
    col1.metric("最优样本数", f"{best_n}", f"HTER={best_hter:.4f}")
    col2.metric("最佳 HTER", f"{best_hter:.4f}")
    col3.metric("对应准确率", f"{float(valid['准确率'].iloc[best_idx]):.4f}")

def _render_model_compare(source):
    st.subheader("⚖️ 模型性能对比")
    st.caption("对比 SVM 与 CNN 的认证性能")
    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.number_input("随机种子", 0, 999, 42, key="cmp_seed")
    with c2:
        if source == "csi":
            na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
            n_files = st.number_input("每用户文件数", na, na * 20, na * 8, na, key="cmp_mf")
        else:
            n_files = st.number_input("每用户文件数", 1, 4, 3, 1, key="cmp_mf")
    with c3:
        test_size = st.slider("测试比例", 0.1, 0.4, 0.2, 0.05, key="cmp_ts")

    op_ui = TrainingOperationUI(start_key="cmp_start", cancel_key="cmp_cancel")
    op_ui.render_controls(start_label="⚡ 开始对比", on_start=SessionStateManager.reset_training)

    is_running = SessionStateManager.is_running()
    has_result = st.session_state.compare_results is not None

    if is_running or has_result:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(
                lambda: ModelCompareExecutor(source, n_files, seed, test_size),
                "compare_results", op_ui
            )
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.compare_results is not None
        if has_result or st.session_state.training_error:
            op_ui.render_results("compare_results")
        if has_result:
            _render_compare_results(st.session_state.compare_results, source)

def _render_compare_results(results, source):
    """论文级模型对比 — 表格 + 分组柱状图含误差线。"""
    if not isinstance(results, dict) or not results:
        st.info("暂无对比结果。")
        return

    st.markdown("---")
    st.subheader("📊 模型对比结果")

    # ── 表格 ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).T
    row_order = [m for m in ["SVM", "CNN"] if m in df.index]
    df = df.loc[row_order]
    df.index.name = "模型"

    has_std = "HTER_std" in df.columns and df["HTER_std"].notna().any()
    tbl_cols = ["HTER", "FAR", "FRR", "准确率"]
    if has_std:
        tbl_cols += ["HTER_std", "FAR_std", "FRR_std"]
    avail = [c for c in tbl_cols if c in df.columns]
    st.dataframe(
        df[avail].style.format(
            {c: "{:.4f}" for c in avail}, na_rep="—",
        ), use_container_width=True,
    )

    if "HTER" not in df.columns or df["HTER"].isna().all():
        return

    # ── 图表 ────────────────────────────────────────────────────────────────
    _setup_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- 左: 错误率对比 ---
    metrics = [("HTER", "#C00000"), ("FAR", "#4472C4"), ("FRR", "#ED7D31")]
    n_models = len(row_order)
    x = np.arange(n_models)
    w = 0.22
    for i, (metric, color) in enumerate(metrics):
        vals = [float(df.loc[m].get(metric, 0) or 0) for m in row_order]
        bars = ax1.bar(x + i * w, vals, w, label=metric, color=color,
                       alpha=0.88, edgecolor='white', lw=0.5, zorder=3)
        # 误差线 (±σ 跨用户标准差)
        std_col = f"{metric}_std"
        if has_std and std_col in df.columns:
            stds = [float(df.loc[m].get(std_col, 0) or 0) for m in row_order]
            if not all(v == 0 for v in stds):
                ax1.errorbar(x + i * w, vals, yerr=stds, fmt='none',
                             ecolor='#333333', capsize=3, lw=1, zorder=4)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=_FS["small"],
                         rotation=90, color='#333333')
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(row_order, fontsize=_FS["normal"])
    ax1.set_ylabel("错误率 (HTER/FAR/FRR)", fontsize=_FS["large"])
   
    ax1.legend(fontsize=_FS["normal"], framealpha=0.85, loc='upper right',
               title="HTER = (FAR+FRR)/2")
    ax1.grid(True, alpha=0.2, axis='y', lw=0.5)
    all_vals = [float(df.loc[m].get(metric, 0) or 0) for m in row_order for metric, _ in metrics]
    ax1.set_ylim(0, max(max(all_vals) * 1.35, 0.3))

    # --- 右: 准确率对比 ---
    acc_vals = [float(df.loc[m].get("准确率", 0) or 0) for m in row_order]
    acc_colors = ["#2E7D32", "#1565C0"]
    bars2 = ax2.bar(row_order, acc_vals, color=[acc_colors[i] for i in range(len(row_order))],
                    alpha=0.88, edgecolor='white', lw=0.5, width=0.45, zorder=3)
    for bar, val in zip(bars2, acc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=_FS["small"],
                 color='#333333')
    ax2.set_ylabel("准确率", fontsize=_FS["large"])
    
    ax2.grid(True, alpha=0.2, axis='y', lw=0.5)
    ax2.set_ylim(0, min(1.0, max(acc_vals) * 1.15) if acc_vals else 1.0)

    source_label = "CSI" if source == "rssi" else "CSI"
    fig.suptitle(f"SVM vs CNN ", fontsize=_FS["title"], fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_experiment_subfigures(fig, "E4")
    st.pyplot(fig); plt.close(fig)

    # ── 最佳模型 ────────────────────────────────────────────────────────────
    best_model = min(row_order, key=lambda m: float(df.loc[m].get("HTER", 1.0) or 1.0))
    best_hter = float(df.loc[best_model, "HTER"])
    col1, col2 = st.columns(2)
    col1.metric("最佳模型", best_model, f"HTER={best_hter:.4f}")
    col2.metric(f"{best_model} 准确率", f"{float(df.loc[best_model, '准确率']):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 认证阶段（单次 + 持续）
# ══════════════════════════════════════════════════════════════════════════════
def _render_single_auth(source, model, pca, scaler, cnn_model, feature_config=None):
    st.subheader("单次认证")

    use_slice = False
    slice_duration = 5.0
    use_combine = False
    uploaded = None
    combine_subject = ""

    if source == "rssi":
        # RSSI: 文件上传 + 可选切片模式
        uploaded = st.file_uploader("上传样本文件", type=["mat"], key="single_auth_rssi")
        if not uploaded:
            st.info("上传 MAT 文件进行认证。")
            return
        c_slice, c_model, c_claim = st.columns(3)
        with c_slice:
            use_slice = st.checkbox("切片模式 (200s → 5s 片段)", False, key="slice_mode",
                                    help="将 200s RSSI 样本切分为 5s 子样本, 每个子样本独立认证")
            if use_slice:
                slice_duration = st.slider("切片时长 (秒)", 1.0, 20.0, 5.0, 1.0, key="slice_dur")
        with c_model:
            mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_rssi")
        with c_claim:
            available_subjects = _get_available_subjects(model, cnn_model, source)
            claimed = st.selectbox("声明身份", available_subjects, key="claim_single_rssi")
    else:
        # CSI: 文件上传 或 组合模式 (拼接所有动作)
        na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
        est_s = na * 5
        use_combine = st.checkbox(
            f"组合模式 ({na} 动作 → ~{est_s}s, 可比肩 RSSI 200s)",
            False, key="csi_combine",
            help=f"将同一用户 {na} 个动作各取 1 个 trial 沿时间轴拼接, 得到与 RSSI 样本规模相当的连续信号")
        if use_combine:
            c_subj, c_trial, c_model, c_claim = st.columns(4)
            with c_subj:
                available_subjects = _get_available_subjects(model, cnn_model, source)
                combine_subject = st.selectbox("用户", available_subjects, key="csi_comb_subj")
            with c_trial:
                combine_trial = st.number_input("试验编号", 1, 20, 1, key="csi_comb_trial")
            with c_model:
                mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_csi_c")
            with c_claim:
                claimed = st.selectbox("声明身份", available_subjects, key="claim_single_csi_c")
        else:
            uploaded = st.file_uploader("上传样本文件", type=["npy"], key="single_auth_csi")
            if not uploaded:
                st.info("上传样本文件或启用组合模式。")
                return
            c1, c2 = st.columns(2)
            with c1:
                mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_csi")
            with c2:
                available_subjects = _get_available_subjects(model, cnn_model, source)
                claimed = st.selectbox("声明身份", available_subjects, key="claim_single_csi")

    if (mtype == "SVM" and model is None) or (mtype == "CNN" and cnn_model is None):
        st.warning(f"{mtype} 模型未训练。")
        return
    if not st.button("开始认证", type="primary", key="btn_single"):
        return

    try:
        # 数据获取: 上传文件 / CSI 组合 / 默认加载
        combine_label = ""
        if source == "csi" and use_combine:
            csi_dir = _CONFIG.npy_dir
            raw = _combine_csi_files(csi_dir, combine_subject, combine_trial)
            if raw is None:
                st.error(f"未找到用户 '{combine_subject}' 的 CSI 文件。")
                return
            combine_label = f" (组合: {combine_subject}, trial={combine_trial}, {raw.shape[0]} 步)"
            st.info(f"已拼接 {raw.shape[0]} 时间步 (~{raw.shape[0]/100:.0f}s) 来自用户 '{combine_subject}'。")
        else:
            raw = _load_upload(uploaded, source)

        if use_slice and source == "rssi":
            # ── 切片模式: 每个 5s 片段独立认证 ──
            slices = _slice_rssi(raw, slice_duration_s=slice_duration)
            st.info(f"Sliced into {len(slices)} segments ({slice_duration}s each). Authenticating...")

            slice_results = []
            all_scores_flat = []
            for i, seg in enumerate(slices):
                windows = _build_windows(seg)
                if windows.shape[0] == 0:
                    continue
                if mtype == "CNN":
                    is_ok, mean_s, scores = cnn_model.predict_authentication(windows, claimed)
                    threshold = cnn_model.thresholds.get(claimed, 0.5)
                    model_label = "1D-CNN"
                else:
                    feats = _extract_features_for_auth(
                        windows, pca, scaler, feature_config,
                        feature_dim=getattr(model, 'feature_dim', None))
                    scores = svm_scores(model.verifiers[claimed], feats)
                    threshold = model.thresholds.get(claimed, 0.5)
                    mean_s = float(np.mean(scores))
                    is_ok = mean_s >= threshold
                    model_label = "SVM (RBF)"
                accept = np.mean(scores >= threshold)
                slice_results.append({"segment": i + 1, "accept_rate": accept,
                                      "mean_score": mean_s, "decision": is_ok,
                                      "n_windows": len(scores)})
                all_scores_flat.extend(scores.tolist())

            # 汇总
            n_accept = sum(1 for r in slice_results if r["decision"])
            n_total = len(slice_results)
            overall_accept_rate = n_accept / max(n_total, 1)
            all_scores_arr = np.array(all_scores_flat)

            decision = "接受" if overall_accept_rate >= 0.5 else "拒绝"
            box_color = "#E8F5E9" if overall_accept_rate >= 0.5 else "#FFEBEE"
            border_color = "#2E7D32" if overall_accept_rate >= 0.5 else "#C62828"
            st.markdown(f"""
            <div style="padding:20px; border-radius:8px; background:{box_color};
                        border:2px solid {border_color}; text-align:center; margin:10px 0;">
                <h2 style="margin:0; color:{border_color};">{decision}</h2>
                <p style="margin:5px 0 0 0; color:#555; font-size:0.95rem;">
                    切片模式 ({slice_duration}s × {n_total} 段) | 模型: {model_label} | 用户: {claimed} |
                    通过段数: {n_accept}/{n_total} ({overall_accept_rate:.1%}) |
                    阈值: {threshold:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 每段分数图
            _setup_paper_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            seg_idx = [r["segment"] for r in slice_results]
            seg_rates = [r["accept_rate"] for r in slice_results]
            colors = ['#2E7D32' if r["decision"] else '#C62828' for r in slice_results]
            ax1.bar(seg_idx, seg_rates, color=colors, alpha=0.8, edgecolor='white')
            ax1.axhline(y=0.5, color='#FF6F00', linestyle='--', linewidth=1.5, label='决策边界 (0.5)')
            ax1.set_xlabel("片段编号", fontsize=_FS["large"])
            ax1.set_ylabel("接受率 (≥阈值窗口占比)", fontsize=_FS["large"])
            ax1.set_title(f"每段接受率 (阈值={threshold:.4f})", fontsize=_FS["large"])
            ax1.set_ylim(0, 1.05)
            ax1.legend(fontsize=_FS["normal"])
            ax1.grid(axis='y', alpha=0.2)
            ax2.hist(all_scores_arr, bins=30, color='#4472C4', alpha=0.7, edgecolor='white')
            ax2.axvline(x=threshold, color='#FF6F00', linestyle='--', linewidth=1.5, label=f'阈值={threshold:.4f}')
            ax2.set_xlabel("认证分数", fontsize=_FS["large"])
            ax2.set_ylabel("窗口数", fontsize=_FS["large"])
            ax2.set_title(f"分数分布 ({len(all_scores_flat)} 窗口)", fontsize=_FS["large"])
            ax2.legend(fontsize=_FS["normal"])
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

            # 汇总统计
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("片段通过率", f"{n_accept}/{n_total} ({overall_accept_rate:.1%})")
            c2.metric("平均分数", f"{float(np.mean(all_scores_arr)):.4f}")
            c3.metric("分数标准差", f"{float(np.std(all_scores_arr)):.4f}")
            c4.metric("总窗口数", str(len(all_scores_arr)))
            c5.metric("平均窗口/段", f"{len(all_scores_arr) / max(n_total, 1):.0f}")
        else:
            # ── 标准模式: 全量样本认证 ──
            windows = _build_windows(raw)
            if windows.shape[0] == 0:
                st.warning("数据长度不足 (需 ≥ 200 时间点)。")
                return
            if mtype == "CNN":
                if claimed not in cnn_model.subjects:
                    st.error(f"用户 '{claimed}' 未注册。")
                    return
                is_ok, mean_s, scores = cnn_model.predict_authentication(windows, claimed)
                threshold = cnn_model.thresholds.get(claimed, 0.5)
                model_label = "1D-CNN"
            else:
                if claimed not in model.verifiers:
                    st.error(f"用户 '{claimed}' 未注册。")
                    return
                feats = _extract_features_for_auth(
                    windows, pca, scaler, feature_config,
                    feature_dim=getattr(model, 'feature_dim', None))
                scores = svm_scores(model.verifiers[claimed], feats)
                threshold = model.thresholds.get(claimed, 0.5)
                mean_s = float(np.mean(scores))
                is_ok = mean_s >= threshold
                model_label = "SVM (RBF)"

            accept_rate = np.mean(scores >= threshold)
            decision = "接受" if is_ok else "拒绝"
            box_color = "#E8F5E9" if is_ok else "#FFEBEE"
            border_color = "#2E7D32" if is_ok else "#C62828"
            st.markdown(f"""
            <div style="padding:20px; border-radius:8px; background:{box_color};
                        border:2px solid {border_color}; text-align:center; margin:10px 0;">
                <h2 style="margin:0; color:{border_color};">{decision}</h2>
                <p style="margin:5px 0 0 0; color:#555; font-size:0.95rem;">
                    模型: {model_label} | 用户: {claimed} |
                    平均分数: {mean_s:.4f} | 阈值: {threshold:.4f} |
                    接受率: {accept_rate:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            _score_chart(scores, threshold, f"认证 — {claimed} ({model_label})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("接受率", f"{accept_rate:.1%}")
            c2.metric("平均分数", f"{mean_s:.4f}")
            c3.metric("分数标准差", f"{np.std(scores):.4f}")
            c4.metric("窗口数", str(len(scores)))
    except Exception as e:
        st.error(f"认证失败: {e}")

def _render_continuous_auth(source, model, pca, scaler, cnn_model, feature_config=None):
    st.subheader("持续认证监控")
    st.caption("实时滑动窗口身份验证。")

    use_combine_cont = False
    combine_subject_cont = None
    combine_trial_cont = 1
    if source == "csi":
        na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
        est_s = na * 5
        use_combine_cont = st.checkbox(
            f"组合模式 ({na} 动作 → ~{est_s}s, 可比肩 RSSI 200s)",
            False, key="csi_combine_cont",
            help=f"将同一用户 {na} 个动作各取 1 个 trial 沿时间轴拼接, 与 RSSI 持续认证可比较")
        if use_combine_cont:
            c_s, c_t = st.columns(2)
            with c_s:
                available_subjects = _get_available_subjects(model, cnn_model, source)
                combine_subject_cont = st.selectbox("用户", available_subjects, key="csi_comb_subj_cont")
            with c_t:
                combine_trial_cont = st.number_input("试验编号", 1, 20, 1, key="csi_comb_trial_cont")

    if not use_combine_cont:
        uploaded = st.file_uploader("上传样本文件", type=["mat"] if source == "rssi" else ["npy"], key="cont_auth")
        if not uploaded:
            st.info("上传样本文件以开始持续监控。")
            return
    c1, c2, c3 = st.columns(3)
    with c1:
        mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_cont")
    with c2:
        available_subjects = _get_available_subjects(model, cnn_model, source)
        claimed = st.selectbox("声明身份", available_subjects, key="claim_cont")
    with c3:
        ws = st.slider("平滑窗口", 3, 30, 10, key="ws_cont")
    if (mtype == "SVM" and model is None) or (mtype == "CNN" and cnn_model is None):
        st.warning(f"{mtype} 模型未训练。")
        return
    if not st.button("开始监控", type="primary", key="btn_cont"):
        return
    try:
        if use_combine_cont and source == "csi":
            raw = _combine_csi_files(_CONFIG.npy_dir, combine_subject_cont, combine_trial_cont)
            if raw is None:
                st.error(f"未找到用户 '{combine_subject_cont}' 的 CSI 文件。")
                return
            st.info(f"已拼接 {raw.shape[0]} 时间步 (~{raw.shape[0]/100:.0f}s) 来自用户 '{combine_subject_cont}'。")
        else:
            raw = _load_upload(uploaded, source)
        windows = _build_windows(raw)
        if windows.shape[0] == 0:
            st.warning("数据长度不足。")
            return
        if mtype == "CNN":
            if claimed not in cnn_model.subjects:
                st.error(f"用户 '{claimed}' 未注册。")
                return
            threshold = cnn_model.thresholds.get(claimed, 0.5)
            all_scores = cnn_model.predict_authentication(windows, claimed)[2]
            model_label = "1D-CNN"
        else:
            if claimed not in model.verifiers:
                st.error(f"用户 '{claimed}' 未注册。")
                return
            feats = _extract_features_for_auth(
                windows, pca, scaler, feature_config,
                feature_dim=getattr(model, 'feature_dim', None))
            all_scores = svm_scores(model.verifiers[claimed], feats)
            threshold = model.thresholds.get(claimed, 0.5)
            model_label = "SVM (RBF)"
        n = len(all_scores)
        smoothed = np.array([np.mean(all_scores[max(0, i - ws + 1):i + 1]) for i in range(n)])
        decisions = smoothed >= threshold

        # 论文级图表: 上 — 分数曲线, 下 — 决策条
        _setup_paper_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                        gridspec_kw={'height_ratios': [4, 1]})
        t = np.arange(n)
        # 绿色/红色背景区分接受/拒绝区域
        ax1.fill_between(t, threshold, 1.0, alpha=0.08, color='#2E7D32')
        ax1.fill_between(t, 0, threshold, alpha=0.08, color='#C62828')
        ax1.plot(t, all_scores, alpha=0.25, color='#9E9E9E', lw=0.5, label='原始分数 (逐窗口)')
        ax1.plot(t, smoothed, color='#1565C0', lw=1.8, label=f'平滑分数 (窗口={ws})')
        ax1.axhline(y=threshold, color='#FF6F00', linestyle='--', lw=1.8,
                    label=f'阈值 = {threshold:.4f}')
        ax1.set_ylabel("认证分数 (0~1, 越高越可信)", fontsize=_FS["large"])
        ax1.set_title(f"持续认证 — 用户 {claimed} ({model_label})", fontsize=_FS["large"])
        ax1.legend(loc='upper right', fontsize=_FS["normal"], framealpha=0.9)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.25)
        # 决策条: 绿色=接受, 红色=拒绝
        colors = ['#2E7D32' if d else '#C62828' for d in decisions]
        ax2.bar(t, np.ones(n), width=1.0, color=colors, alpha=0.7)
        ax2.set_xlabel("窗口序号", fontsize=_FS["large"])
        ax2.set_ylabel("决策", fontsize=_FS["large"])
        ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # 统计指标
        longest = streak = 0
        for d in decisions:
            if d: streak += 1; longest = max(longest, streak)
            else: streak = 0
        switches = sum(1 for i in range(1, n) if decisions[i] != decisions[i-1])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("接受率", f"{np.mean(decisions):.1%}")
        c2.metric("最终决策", "接受" if decisions[-1] else "拒绝")
        c3.metric("最长连续接受", f"{longest} 窗口")
        c4.metric("状态切换次数", str(switches))
        c5.metric("总窗口数", str(n))
        if switches > 2:
            st.warning(f"检测到 {switches} 次状态切换 — 可能存在身份变化。")
    except Exception as e:
        st.error(f"持续认证失败: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════
def _render_experiments(source):
    st.subheader("🧪 实验运行面板")
    st.caption("一键运行论文实验 (E1~E4)，结果以图表与表格呈现。")

    # 根据数据源过滤实验选项
    all_exps = {
        "E1: CSI单次认证": {"key": "E1", "sources": ("csi",)},
        "E2: RSSI持续认证(仅RSSI)": {"key": "E2", "sources": ("rssi",)},
        "E3: 注册时长影响(仅CSI)": {"key": "E3", "sources": ("csi",)},
        "E4: 模型对比": {"key": "E4", "sources": ("csi", "rssi")},
        "E5: RSSI切片单次认证(仅RSSI)": {"key": "E5", "sources": ("rssi",)},
    }
    exp_options = list(all_exps.keys())
    exp_defaults = [o for o in exp_options if source in all_exps[o]["sources"]]
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input("随机种子", 0, 999, 42, key="exp_seed")
    with c2:
        exp_select = st.multiselect(
            "选择实验",
            exp_options,
            default=exp_defaults[:2],
            key="exp_select",
            help=f"当前数据源: {source.upper()}。仅显示兼容的实验。")
    # 标记不兼容的实验
    incompatible = [e for e in exp_select if source not in all_exps[e]["sources"]]
    if incompatible:
        st.warning(f"以下实验与当前数据源 ({source.upper()}) 不兼容, 将被跳过: "
                   f"{', '.join(incompatible)}")

    op_ui = TrainingOperationUI(start_key="exp_start", cancel_key="exp_cancel")
    def _on_exp_start():
        if not exp_select:
            st.warning("请至少选择一个实验")
            st.stop()
        valid = [e for e in exp_select if source in all_exps[e]["sources"]]
        if not valid:
            st.error(f"所选实验均与当前数据源 ({source.upper()}) 不兼容, 无法运行。")
            st.stop()
        SessionStateManager.reset_training()
    op_ui.render_controls(start_label="▶️ 运行实验", on_start=_on_exp_start)

    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("exp_results") is not None

    if is_running or has_result:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(
                lambda: _ExperimentWrapper(source, exp_select),
                "exp_results", op_ui)
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("exp_results") is not None
        if has_result or st.session_state.training_error:
            op_ui.render_results("exp_results")
        if has_result:
            from experiments import (
                render_e1_results,
                render_e2_results,
                render_e3_results,
                render_e5_results,
            )
            res = st.session_state.exp_results
            if "E1" in res: render_e1_results(res["E1"])
            if "E2" in res: render_e2_results(res["E2"])
            if "E3" in res: render_e3_results(res["E3"])
            if "E4" in res: _render_compare_results(res["E4"], "")
            if "E5" in res: render_e5_results(res["E5"])


class _ExperimentWrapper(TrainingExecutor):
    """实验包装器 — 兼容 TrainingExecutor 接口。"""
    def __init__(self, source, exp_select):
        super().__init__("实验", source)
        self.exp_select = exp_select

    def get_config(self):
        return {
            "data_source": self.source,
            "experiments": self.exp_select,
        }

    def _execute_core(self):
        from experiments import ExperimentRunner
        runner = ExperimentRunner(self.source, check_cancelled=_check_cancelled)
        exp_keys = set()
        skipped = []
        for e in self.exp_select:
            if e.startswith("E1"): exp_keys.add("E1")
            elif e.startswith("E2"):
                if self.source == "rssi": exp_keys.add("E2")
                else: skipped.append(e)
            elif e.startswith("E3"):
                if self.source == "csi": exp_keys.add("E3")
                else: skipped.append(e)
            elif e.startswith("E4"): exp_keys.add("E4")
            elif e.startswith("E5"):
                if self.source == "rssi": exp_keys.add("E5")
                else: skipped.append(e)
        if not exp_keys:
            reason = f"数据源为 {self.source.upper()}"
            if skipped:
                reason += f"，{', '.join(skipped)} 与该数据源不兼容"
            raise RuntimeError(f"无可运行的实验: {reason}")
        if skipped:
            self.logger.info(f"跳过不兼容实验: {', '.join(skipped)}")
        total = len(exp_keys)
        for i, ek in enumerate(sorted(exp_keys)):
            _check_cancelled()
            progress = (i + 0.1) / max(total, 1)
            self._update_ui(progress, f"运行 {ek}...",
                            "training-running", f"🔄 实验: {ek}")
            if ek == "E1": runner.run_e1()
            elif ek == "E2": runner.run_e2()
            elif ek == "E3": runner.run_e3()
            elif ek == "E4": runner.run_e4()
            elif ek == "E5": runner.run_e5()
            self._update_ui((i + 1) / max(total, 1), f"{ek} 完成",
                            "training-running", f"✅ {ek}")
        st.session_state.exp_results = runner.results
        return runner.results

    def _on_success(self, result):
        """从嵌套实验结果中提取最佳 HTER 用于日志摘要。"""
        self._update_ui(1.0, "训练完成！", "training-completed", "✅ 实验完成！")
        hter = "N/A"
        if isinstance(result, dict):
            for exp_id, exp_data in result.items():
                if exp_id == "E3" and isinstance(exp_data, list):
                    valid = [r for r in exp_data if r.get("HTER") is not None]
                    if valid:
                        hter = f"{min(r['HTER'] for r in valid):.4f} (E3 best)"
                elif isinstance(exp_data, dict):
                    for k, v in exp_data.items():
                        if isinstance(v, dict) and "mean_hter" in v:
                            hter = f"{v['mean_hter']:.4f} ({exp_id}/{k})"
                            break
        self.logger.info(f"实验完成 - HTER: {hter}")


def main():
    SessionStateManager.init()

    st.markdown('<div class="auth-header">🔐 身份认证系统</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 阶段")
        phase = st.radio("选择阶段", ["📝 注册阶段 (训练)", "🔍 认证阶段 (推理)"], key="phase")
        st.markdown("---")
        st.markdown("## 数据源")
        source_label = st.radio("选择数据源", ["📡 静态 (RSSI/MAT)", "📶 动态 (CSI/NPY)"], key="source")
        st.markdown("---")
        st.markdown("## 系统信息")
        st.markdown(f'<div class="memory-info">系统内存: {_memory_monitor.system_memory_pressure:.0%}<br>GPU内存: {_memory_monitor.gpu_memory_pressure:.0%}<br>项目: <code>{_CONFIG.root_dir}</code><br></div>', unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("💾 缓存管理", expanded=False):
            cache = st.session_state.pipeline_cache
            if cache:
                st.write(f"缓存项: {len(cache._cache_index)}")
                if st.button("🗑️ 清空所有缓存", key="clear_cache"):
                    cache.clear()
                    st.success("缓存已清空")
                    st.rerun()

    source: DataSource = "rssi" if "静态" in source_label else "csi"
    is_reg = "注册" in phase

    st.markdown(f'<div class="phase-label">{"📝 注册阶段 — 模型训练" if is_reg else "🔍 认证阶段 — 模型推理"}</div>', unsafe_allow_html=True)

    if is_reg:
        ensure_server()
        log_html = write_log_html(_CONFIG.cache_dir)
        st.iframe(src=str(log_html), height=340)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔧 基础训练", "📈 参数研究", "⚖️ 模型对比", "🧪 实验"])
        with tab1:
            _render_register_basic(source)
        with tab2:
            _render_param_study(source)
        with tab3:
            _render_model_compare(source)
        with tab4:
            _render_experiments(source)
    else:
        # 从注册阶段切回时清除模型缓存, 确保加载最新训练的模型
        if st.session_state.get("_last_phase", "") != "inference":
            _load_auth_model.clear()
            _load_cnn_model.clear()
            st.session_state["_last_phase"] = "inference"
        model, pca, scaler, feature_config = _load_auth_model(source)
        cnn_model = _load_cnn_model(source)
        tab1, tab2 = st.tabs(["📤 单次认证", "⏱️ 持续认证"])
        with tab1:
            _render_single_auth(source, model, pca, scaler, cnn_model, feature_config)
        with tab2:
            _render_continuous_auth(source, model, pca, scaler, cnn_model, feature_config)


if __name__ == "__main__":
    main()