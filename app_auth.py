# -*- coding: utf-8 -*-
"""身份认证交互式系统 — 注册阶段 + 认证阶段。

模块化架构:
  - app.state     → Session 状态管理 + 全局配置
  - app.ui        → UI 组件 (按钮、进度条、字体)
  - app.auth      → 认证工具 (模型加载、CSI 拼接、分数图表)
  - app.executor  → 训练执行器 (基础/实验包装器)
"""
from __future__ import annotations

import gc
import threading
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.app_utils import (
    FONT_SIZES as _FS,
    build_windows as _build_windows,
    extract_features_for_auth as _extract_features_for_auth,
    setup_paper_style as _setup_paper_style,
    slice_rssi as _slice_rssi,
)
from scripts.models import clear_gpu_memory, log_training, svm_scores
from scripts.pipeline_runner import AuthPipeline
from scripts.log_server import ensure_server, write_log_html

from app.state import SessionStateManager, DataSource, _CONFIG, _memory_monitor
from app.ui import TrainingCancelledError, TrainingOperationUI, check_cancelled
from app.auth import (
    combine_csi_files as _combine_csi_files,
    estimate_memory as _estimate_memory_usage,
    get_available_subjects as _get_available_subjects,
    load_cnn_model as _load_cnn_model,
    load_svm_model as _load_auth_model,
    load_uploaded as _load_upload,
    normalize_model_keys as _normalize_model_keys,
    score_chart as _score_chart,
    show_metrics as _show_metrics,
    show_model_status as _show_model_status,
)
from app.executor import BasicTrainingExecutor, TrainingExecutor, _ExperimentWrapper

warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message="resource_tracker")

# ══════════════════════════════════════════════════════════════════════════════
# ── ParamStudyExecutor (因耦合度较高, 保留在 app_auth.py 中)
# ══════════════════════════════════════════════════════════════════════════════

class ParamStudyExecutor(TrainingExecutor):
    """参数研究 — 遍历不同样本数量进行训练对比。

    优化策略:
      1. 全量特征仅准备一次 → memmap 落盘 → OS 按需分页
      2. 在线 SVM (SGD+RBFSampler) 固定参数, 跳过超参搜索
      3. 子采样使用索引数组代替数据复制 → 零额外内存
      4. finally 块确保 memmap 清理
    """

    def __init__(self, source, mtype, study_sizes, seed):
        super().__init__(f"{mtype} 参数研究", source)
        self._model = mtype
        self.study_sizes = sorted(study_sizes)
        self.seed = seed
        self._mmap_paths: list[Path] = []

    def get_config(self):
        return {"task": "param_study", "study_sizes": self.study_sizes, "seed": self.seed}

    @staticmethod
    def _stratified_indices(y, ratio, seed):
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

    def _prepare_memmap(self, max_files):
        pipeline = AuthPipeline(
            data_source=self.source, seed=self.seed, test_size=0.2,
            use_pca=False, max_files_per_subject=max_files,
            use_cache=True, use_model_cache=False,
            save_model=False, save_metrics=False, clean_intermediate=False,
            csi_denoise=_CONFIG.csi_denoise,
        )
        prepared = pipeline._prepare()
        x_tr = np.asarray(prepared[0], dtype=np.float32)
        y_tr = list(prepared[1]) if not isinstance(prepared[1], np.ndarray) else prepared[1].tolist()
        x_te = np.asarray(prepared[2], dtype=np.float32)
        y_te = list(prepared[3]) if not isinstance(prepared[3], np.ndarray) else prepared[3].tolist()
        tr_shape, te_shape = x_tr.shape, x_te.shape
        pca_model = getattr(pipeline._fe, 'pca', None) if hasattr(pipeline, '_fe') else None
        scaler_model = getattr(pipeline._fe, 'scaler', None) if hasattr(pipeline, '_fe') else None
        del prepared, pipeline; gc.collect()

        mmap_dir = _CONFIG.cache_dir / "param_study_mmap"
        mmap_dir.mkdir(parents=True, exist_ok=True)
        for old in mmap_dir.glob("*.dat"):
            old.unlink(missing_ok=True)
        xt_path = mmap_dir / "x_train.dat"
        xe_path = mmap_dir / "x_test.dat"
        self._mmap_paths = [xt_path, xe_path]
        self.logger.info(f"memmap: train {tr_shape}, test {te_shape}")
        xt_mmap = np.memmap(xt_path, dtype=np.float32, mode='w+', shape=tr_shape)
        xt_mmap[:] = x_tr[:]; xt_mmap.flush()
        xe_mmap = np.memmap(xe_path, dtype=np.float32, mode='w+', shape=te_shape)
        xe_mmap[:] = x_te[:]; xe_mmap.flush()
        del x_tr, x_te, xt_mmap, xe_mmap; gc.collect()
        return {"xt_path": xt_path, "tr_shape": tr_shape, "y_train": y_tr,
                "xe_path": xe_path, "te_shape": te_shape, "y_test": y_te,
                "pca_model": pca_model, "scaler_model": scaler_model}

    def _execute_core(self):
        results = []
        total = len(self.study_sizes)
        max_files = max(self.study_sizes)
        if self._model != "SVM":
            for i, n_files in enumerate(self.study_sizes):
                check_cancelled()
                results.append(self._run_single_study(n_files, i, total))
            st.session_state.study_results = results
            return results
        mm = None
        try:
            self._update_ui(0.05, f"提取特征 → memmap (max={max_files})...",
                           "training-running", "🔄 参数研究: memmap 准备...")
            mm = self._prepare_memmap(max_files)
            x_tr_full = np.memmap(mm["xt_path"], dtype=np.float32, mode='r', shape=mm["tr_shape"])
            x_te_full = np.memmap(mm["xe_path"], dtype=np.float32, mode='r', shape=mm["te_shape"])
            y_tr_full = np.asarray(mm["y_train"])
            y_te_full = np.asarray(mm["y_test"])
            self._update_ui(0.10, "准备在线 SVM 训练...", "training-running", "🔄 参数研究: 在线SVM准备...")
            best_params = {"alpha": 1e-4, "gamma": "scale", "n_components": 200}
            rank = 1
            for n_files in sorted(self.study_sizes, reverse=True):
                check_cancelled()
                progress = 0.12 + rank / max(total, 1) * 0.83
                self._update_ui(progress, f"训练: {n_files} files ({rank}/{total})",
                               "training-running", f"🔄 研究: {rank}/{total}")
                ratio = n_files / max_files
                if ratio >= 1.0:
                    indices = np.arange(len(y_tr_full), dtype=np.intp)
                    y_tr_sub = y_tr_full
                else:
                    indices, y_tr_sub = self._stratified_indices(y_tr_full, ratio, seed=self.seed + n_files)
                entry = self._train_with_data(
                    x_tr_full, indices, y_tr_sub, x_te_full, y_te_full,
                    best_params, n_files, rank, total,
                    pca_model=mm.get("pca_model"), scaler_model=mm.get("scaler_model"))
                results.append(entry)
                st.session_state.study_results = list(results)
                clear_gpu_memory(); gc.collect()
                rank += 1
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"参数研究失败: {e}")
            traceback.print_exc()
            for i, n_files in enumerate(self.study_sizes):
                check_cancelled()
                results.append(self._run_single_study(n_files, i, total))
        finally:
            for p in self._mmap_paths:
                try: p.unlink(missing_ok=True)
                except Exception: pass
            self._mmap_paths.clear()
        results.reverse()
        self.study_sizes.sort()
        st.session_state.study_results = results
        return results

    def _train_with_data(self, x_tr_full, indices, y_tr_sub, x_te, y_te,
                         best_params, n_files, idx, total, pca_model=None, scaler_model=None):
        try:
            from scripts.models import SVMConfig, SVMAuthenticationTrainer
            sc = SVMConfig(threshold_method="youden", random_seed=self.seed, cv_folds=3)
            trainer = SVMAuthenticationTrainer(sc, use_online=True, online_kernel="rbf")
            x_tr_sub = np.asarray(x_tr_full[indices], dtype=np.float32)
            result = trainer.train_from_arrays(
                x_tr_sub, y_tr_sub, x_te, y_te,
                pca_model=pca_model, scaler_model=scaler_model, data_source=self.source)
            sm = result.get("system_metrics", {})
            entry = {"每用户文件数": n_files, "HTER": sm.get("mean_hter", 0),
                     "FAR": sm.get("mean_far", 0), "FRR": sm.get("mean_frr", 0),
                     "准确率": sm.get("global_accuracy", 0),
                     "HTER_std": sm.get("std_hter", 0),
                     "FAR_std": sm.get("std_far", 0), "FRR_std": sm.get("std_frr", 0)}
            self.logger.info(f"✓ [{idx}/{total}]: {n_files} files → HTER={sm.get('mean_hter', 'N/A'):.4f}")
            return entry
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"✗ [{idx}/{total}] ({n_files}): {e}")
            return {"每用户文件数": n_files, "HTER": None, "FAR": None,
                    "FRR": None, "准确率": None, "HTER_std": None,
                    "FAR_std": None, "FRR_std": None, "错误": str(e)}

    def _run_single_study(self, n_files, index, total):
        try:
            pipeline = AuthPipeline(
                data_source=self.source, seed=self.seed, test_size=0.2,
                use_pca=False, max_files_per_subject=n_files,
                use_cache=True, use_model_cache=False,
                save_model=False, save_metrics=False, clean_intermediate=False,
                csi_denoise=_CONFIG.csi_denoise)
            if self._model == "SVM":
                result = pipeline.run_svm()
            else:
                result = pipeline.run_cnn(epochs=10, batch_size=64, cancel_fn=check_cancelled)
            sm = result.get("system_metrics", {})
            entry = {"每用户文件数": n_files, "HTER": sm.get("mean_hter", 0),
                     "FAR": sm.get("mean_far", 0), "FRR": sm.get("mean_frr", 0),
                     "准确率": sm.get("global_accuracy", 0),
                     "HTER_std": sm.get("std_hter", 0),
                     "FAR_std": sm.get("std_far", 0), "FRR_std": sm.get("std_frr", 0)}
            self.logger.info(f"✓ [{index+1}/{total}]: {n_files} files → HTER={sm.get('mean_hter', 'N/A'):.4f}")
            return entry
        except TrainingCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"✗ [{index+1}/{total}] ({n_files}): {e}")
            return {"每用户文件数": n_files, "HTER": None, "FAR": None,
                    "FRR": None, "准确率": None, "HTER_std": None,
                    "FAR_std": None, "FRR_std": None, "错误": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# ── ModelCompareExecutor
# ══════════════════════════════════════════════════════════════════════════════

class ModelCompareExecutor(TrainingExecutor):
    def __init__(self, source, n_files, seed, test_size):
        super().__init__("模型对比", source)
        self.n_files = n_files
        self.seed = seed
        self.test_size = test_size

    def get_config(self):
        return {"task": "model_compare", "n_files": self.n_files,
                "seed": self.seed, "test_size": self.test_size}

    def _execute_core(self):
        results = {}
        check_cancelled()
        self._update_ui(0.1, "训练 SVM (1/2)...", "training-running", "🔄 模型对比: SVM (1/2)...")
        results["SVM"] = self._train_svm()
        clear_gpu_memory(); gc.collect()
        check_cancelled()
        self._update_ui(0.5, "训练 CNN (2/2)...", "training-running", "🔄 模型对比: CNN (2/2)...")
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
        t0 = time.time()
        p = self._make_pipeline(use_online_svm=True, online_kernel="linear",
                                threshold_method="youden", cv_folds=5)
        r = p.run_svm()
        sm = r["system_metrics"]
        dur = time.time() - t0
        return {"HTER": sm["mean_hter"], "FAR": sm["mean_far"],
                "FRR": sm["mean_frr"], "准确率": sm["global_accuracy"],
                "HTER_std": sm.get("std_hter", 0), "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0), "训练耗时(s)": round(dur, 1)}

    def _train_cnn(self):
        t0 = time.time()
        from experiments.base import BaseExperimentRunner
        runner = BaseExperimentRunner.__new__(BaseExperimentRunner)
        runner.source = self.source
        runner._pcfg = _CONFIG
        runner._check_cancelled = check_cancelled
        runner.logger = self.logger
        r = runner._run_cnn(
            epochs=10, batch_size=96, max_files_per_subject=self.n_files,
            use_checkpoint=False, gradient_accumulation_steps=1,
            use_cache=True, use_model_cache=True,
            conv_channels=(32, 64, 128, 192), hidden_units=256)
        sm = r["system_metrics"]
        dur = time.time() - t0
        return {"HTER": sm["mean_hter"], "FAR": sm["mean_far"],
                "FRR": sm["mean_frr"], "准确率": sm["global_accuracy"],
                "HTER_std": sm.get("std_hter", 0), "FAR_std": sm.get("std_far", 0),
                "FRR_std": sm.get("std_frr", 0), "训练耗时(s)": round(dur, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# ── 训练启动 (防重入)
# ══════════════════════════════════════════════════════════════════════════════

_training_lock = threading.Lock()

def _launch_training(executor_factory, result_key, op_ui=None):
    if st.session_state.get(result_key) is not None:
        return
    if not _training_lock.acquire(blocking=False):
        return
    try:
        t_start = time.time()
        executor = executor_factory()
        executor._op_ui = op_ui
        result = executor.run()
        if result is not None:
            st.session_state[result_key] = result
        metrics = {}
        if isinstance(result, dict):
            if "system_metrics" in result:
                metrics = result["system_metrics"]
            elif any(k.startswith("E") and k[1:].isdigit() for k in result):
                best_hter = 1.0
                for exp_data in result.values():
                    if isinstance(exp_data, dict):
                        for v in exp_data.values():
                            if isinstance(v, dict):
                                h = v.get("mean_hter", v.get("HTER"))
                                if h is not None: best_hter = min(best_hter, float(h))
                    elif isinstance(exp_data, list):
                        for r in exp_data:
                            h = r.get("HTER")
                            if h is not None: best_hter = min(best_hter, float(h))
                metrics = {"mean_hter": best_hter if best_hter < 1.0 else "N/A"}
        elif isinstance(result, list) and result:
            best = min((r for r in result if r.get("HTER") is not None),
                      key=lambda r: r["HTER"], default={})
            metrics = {"mean_hter": best.get("HTER", "N/A")}
        log_training(pipeline_config=_CONFIG, model_type=executor.mtype,
                     task_type="authentication", data_source=executor.source,
                     status="success", duration=time.time() - t_start,
                     config=executor.get_config(), metrics=metrics)
    except TrainingCancelledError:
        if executor:
            log_training(pipeline_config=_CONFIG, model_type=executor.mtype,
                         task_type="authentication", data_source=executor.source,
                         status="cancelled", duration=time.time() - t_start,
                         config=executor.get_config())
    except Exception as e:
        err_msg = str(e)
        mtype = executor.mtype if executor else "unknown"
        src = executor.source if executor else "rssi"
        cfg = executor.get_config() if executor else {}
        log_training(pipeline_config=_CONFIG, model_type=mtype,
                     task_type="authentication", data_source=src,
                     status="failed", duration=time.time() - t_start,
                     config=cfg, error=err_msg[:500])
        st.session_state.training_error = err_msg
    finally:
        _training_lock.release()


# ══════════════════════════════════════════════════════════════════════════════
# ── 注册阶段渲染
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
        feature_groups = st.multiselect("特征组", ["spectral", "statistical", "temporal"],
                                        default=["spectral", "statistical", "temporal"], key="reg_fg")
        use_cache = st.checkbox("启用缓存", True, key="reg_cache")
        clean_intermediate = st.checkbox("清理中间文件", True, key="reg_clean")
        csi_denoise = None
        if source == "csi":
            csi_denoise = st.selectbox("CSI 降噪方法", ["无", "hampel", "savgol", "butterworth"], key="reg_denoise")
            csi_denoise = None if csi_denoise == "无" else csi_denoise
        use_online_svm, online_kernel = False, "linear"
        if mtype == "SVM" and source == "csi":
            use_online_svm = st.checkbox("在线SVM (SGD, 增量学习)", False, key="reg_online")
            if use_online_svm:
                online_kernel = st.selectbox("在线SVM核函数", ["linear", "rbf"], key="reg_online_kernel")
    extra = {}
    if mtype == "CNN":
        ec1, ec2, ec3 = st.columns(3)
        with ec1: extra["epochs"] = st.number_input("Epochs", 5, 100, 20, 5, key="reg_ep")
        with ec2: extra["batch_size"] = st.number_input("Batch", 8, 256, 64, 8, key="reg_bs")
        with ec3: extra["learning_rate"] = st.number_input("LR", 1e-5, 0.1, 1e-3, format="%.5f", key="reg_lr")
    kw = dict(data_source=source, seed=seed, test_size=test_size, window_size=ws,
              step_size=ss, use_pca=use_pca, threshold_method=threshold_method,
              use_cache=use_cache, clean_intermediate=clean_intermediate,
              max_files_per_subject=max_files, use_online_svm=use_online_svm,
              online_kernel=online_kernel, feature_groups=tuple(feature_groups),
              csi_denoise=csi_denoise)
    mem_info = _estimate_memory_usage(source, **kw)
    with st.expander("💾 内存评估", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("原始数据", mem_info["原始数据"]); mc2.metric("窗口数据", mem_info["窗口数据"])
        mc3.metric("优化后", mem_info["优化后预估"])
        st.caption(f"样本数: {mem_info['样本数']} | 窗口数: {mem_info['预估窗口数']:,} | 建议批次: {mem_info['建议批次']}")
    op_ui = TrainingOperationUI(start_key="reg_start", cancel_key="reg_cancel")
    op_ui.render_controls(start_label="🚀 开始注册训练", on_start=SessionStateManager.reset_training)
    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("training_result") is not None
    has_error = st.session_state.training_error is not None
    if is_running or has_result or has_error:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(lambda: BasicTrainingExecutor(source, mtype, dict(kw), extra), "training_result", op_ui)
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("training_result") is not None
        if has_result or has_error:
            op_ui.render_results()
        if has_result:
            _show_metrics(st.session_state.training_result.get("system_metrics", {}))


def _render_param_study(source):
    st.subheader("📈 注册参数研究")
    sizes = _CONFIG.PARAM_STUDY_DEFAULTS[source]["sizes"]
    defaults = _CONFIG.PARAM_STUDY_DEFAULTS[source]["default"]
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input("随机种子", 0, 999, 42, key="study_seed")
        study_mtype = st.selectbox("模型", ["SVM", "CNN"], key="study_mtype")
    with c2:
        study_sizes = [int(x) for x in st.multiselect("样本数量", sizes, default=defaults, key="study_sizes",
                                                        help=f"每用户文件数: {sizes}")]
    if not study_sizes:
        st.info("请选择至少一个样本数量")
        return
    op_ui = TrainingOperationUI(start_key="study_start", cancel_key="study_cancel")
    op_ui.render_controls(start_label="▶️ 运行研究", on_start=SessionStateManager.reset_training)
    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("study_results") is not None
    if is_running or has_result:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(lambda: ParamStudyExecutor(source, study_mtype, study_sizes, seed), "study_results", op_ui)
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("study_results") is not None
        if has_result or st.session_state.training_error:
            op_ui.render_results("study_results")
        if has_result:
            _render_study_results(st.session_state.study_results, study_mtype)


def _render_study_results(results, mtype):
    if not results or not isinstance(results, list):
        return
    valid = [r for r in results if r.get("HTER") is not None]
    if not valid:
        return
    st.markdown("---")
    st.subheader("📊 参数研究结果")
    df = pd.DataFrame(results)
    st.dataframe(df.style.format({"HTER": "{:.4f}", "FAR": "{:.4f}", "FRR": "{:.4f}", "准确率": "{:.4f}"}),
                 hide_index=True, use_container_width=True)
    _setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df["每用户文件数"].values.astype(float)
    for metric, color, marker in [("HTER", "#C00000", "o"), ("FAR", "#4472C4", "s"), ("FRR", "#ED7D31", "^")]:
        y = df[metric].values.astype(float)
        ax1.plot(x, y, marker=marker, color=color, lw=2, ms=8, label=metric)
        std_col = f"{metric}_std"
        if std_col in df.columns:
            std_v = df[std_col].values.astype(float)
            if not np.all(std_v == 0):
                ax1.fill_between(x, np.maximum(0, y - std_v), y + std_v, color=color, alpha=0.12)
    ax1.set_xlabel("训练样本 / 用户", fontsize=_FS["large"])
    ax1.set_ylabel("错误率 (HTER/FAR/FRR)", fontsize=_FS["large"])
    ax1.set_title("(a) 错误率随样本数变化", fontsize=_FS["large"]); ax1.legend(fontsize=_FS["normal"])
    ax1.grid(alpha=0.2)
    acc = df["准确率"].values.astype(float)
    ax2.plot(x, acc, 'D-', color='#2E7D32', lw=2, ms=8)
    ax2.set_xlabel("训练样本 / 用户", fontsize=_FS["large"]); ax2.set_ylabel("准确率", fontsize=_FS["large"])
    ax2.set_title("(b) 准确率随样本数变化", fontsize=_FS["large"]); ax2.grid(alpha=0.2)
    fig.suptitle(f"参数研究 — {mtype}", fontsize=_FS["title"], fontweight='bold', y=1.01)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)


def _render_model_compare(source):
    st.subheader("⚖️ 模型对比")
    c1, c2, c3 = st.columns(3)
    na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
    from scripts.config import Defaults
    with c1: seed = st.number_input("随机种子", 0, 999, 42, key="cmp_seed")
    with c2: n_files = st.number_input("每用户文件数", na, na * 20, na * 8, na, key="cmp_files")
    with c3: ts = st.slider("测试比例", 0.1, 0.4, 0.2, 0.05, key="cmp_ts")
    if st.button("开始对比", type="primary", key="cmp_start"):
        SessionStateManager.reset_training()
    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("compare_results") is not None
    can_run = is_running and not SessionStateManager.is_launched() and not has_result
    if is_running or has_result:
        if can_run:
            _launch_training(lambda: ModelCompareExecutor(source, n_files, seed, ts), "compare_results")
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("compare_results") is not None
        if has_result or st.session_state.training_error:
            op_ui = TrainingOperationUI(start_key="cmp_start_main", cancel_key="cmp_cancel")
            op_ui.render_results("compare_results")
        if has_result:
            _render_compare_results(st.session_state.compare_results, source)


def _render_compare_results(results, source):
    if not isinstance(results, dict) or not results:
        return
    st.markdown("---")
    st.subheader("📊 模型对比结果")
    df_data = {}
    for model_name, m in results.items():
        if isinstance(m, dict):
            df_data[model_name] = {k: v for k, v in m.items()}
    if not df_data:
        return
    df = pd.DataFrame(df_data).T
    st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if "HTER" in c or "FAR" in c or "FRR" in c or "准确率" in c}),
                 use_container_width=True)
    _setup_paper_style()
    row_order = list(df_data.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    metrics = [("HTER", "#C00000"), ("FAR", "#4472C4"), ("FRR", "#ED7D31")]
    n_metrics, n_models = len(metrics), len(row_order)
    x = np.arange(n_metrics)
    w = 0.35 / n_models
    for i, model in enumerate(row_order):
        vals = [float(df.loc[model].get(m, 0) or 0) for m, _ in metrics]
        stds = [float(df.loc[model].get(f"{m}_std", 0) or 0) for m, _ in metrics]
        bars = ax1.bar(x + i * w, vals, w, label=model,
                       color=["#2E7D32", "#1565C0"][i], alpha=0.85, edgecolor="white")
        if not all(v == 0 for v in stds):
            ax1.errorbar(x + i * w, vals, yerr=stds, fmt='none', ecolor='#333333', capsize=3, lw=1)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=_FS["small"], rotation=90, color='#333333')
    ax1.set_xticks(x + w); ax1.set_xticklabels([m for m, _ in metrics], fontsize=_FS["normal"])
    ax1.set_ylabel("错误率 (HTER/FAR/FRR)", fontsize=_FS["large"])
    ax1.set_title("(a) 错误率对比 (含 ±σ 跨用户标准差)", fontsize=_FS["large"])
    ax1.legend(fontsize=_FS["normal"], framealpha=0.85, loc='upper right')
    ax1.grid(True, alpha=0.2, axis='y', lw=0.5)
    all_vals = [float(df.loc[m].get(metric, 0) or 0) for m in row_order for metric, _ in metrics]
    ax1.set_ylim(0, max(max(all_vals) * 1.35, 0.3))
    acc_vals = [float(df.loc[m].get("准确率", 0) or 0) for m in row_order]
    ax2.bar(row_order, acc_vals, color=["#2E7D32", "#1565C0"], alpha=0.88, edgecolor="white", width=0.45)
    for bar, val in zip(ax2.patches, acc_vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=_FS["small"], color='#333333')
    ax2.set_ylabel("准确率", fontsize=_FS["large"])
    ax2.set_title("(b) 准确率对比", fontsize=_FS["large"])
    ax2.grid(True, alpha=0.2, axis='y', lw=0.5)
    ax2.set_ylim(0, min(1.0, max(acc_vals) * 1.15) if acc_vals else 1.0)
    fig.suptitle(f"SVM vs CNN ", fontsize=_FS["title"], fontweight='bold', y=1.01)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ── 认证阶段渲染
# ══════════════════════════════════════════════════════════════════════════════

def _render_single_auth(source, model, pca, scaler, cnn_model, feature_config=None):
    st.subheader("单次认证")
    use_slice, slice_duration = False, 5.0
    use_combine, uploaded, combine_subject = False, None, ""
    if source == "rssi":
        uploaded = st.file_uploader("上传样本文件", type=["mat"], key="single_auth_rssi")
        if not uploaded:
            st.info("上传 MAT 文件进行认证。"); return
        c_slice, c_model, c_claim = st.columns(3)
        with c_slice:
            use_slice = st.checkbox("切片模式 (200s → 5s 片段)", False, key="slice_mode")
            if use_slice: slice_duration = st.slider("切片时长 (秒)", 1.0, 20.0, 5.0, 1.0, key="slice_dur")
        with c_model: mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_rssi")
        with c_claim:
            claimed = st.selectbox("声明身份", _get_available_subjects(model, cnn_model, source), key="claim_single_rssi")
    else:
        na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
        use_combine = st.checkbox(f"组合模式 ({na} 动作 → ~{na*5}s, 可比肩 RSSI 200s)", False, key="csi_combine")
        if use_combine:
            c_subj, c_trial, c_model, c_claim = st.columns(4)
            with c_subj: combine_subject = st.selectbox("用户", _get_available_subjects(model, cnn_model, source), key="csi_comb_subj")
            with c_trial: combine_trial = st.number_input("试验编号", 1, 20, 1, key="csi_comb_trial")
            with c_model: mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_csi_c")
            with c_claim: claimed = st.selectbox("声明身份", _get_available_subjects(model, cnn_model, source), key="claim_single_csi_c")
        else:
            uploaded = st.file_uploader("上传样本文件", type=["npy"], key="single_auth_csi")
            if not uploaded: st.info("上传样本文件或启用组合模式。"); return
            c1, c2 = st.columns(2)
            with c1: mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_single_csi")
            with c2: claimed = st.selectbox("声明身份", _get_available_subjects(model, cnn_model, source), key="claim_single_csi")
    if (mtype == "SVM" and model is None) or (mtype == "CNN" and cnn_model is None):
        st.warning(f"{mtype} 模型未训练。"); return
    if not st.button("开始认证", type="primary", key="btn_single"): return
    try:
        if source == "csi" and use_combine:
            raw = _combine_csi_files(_CONFIG.npy_dir, combine_subject, combine_trial)
            if raw is None: st.error(f"未找到用户 '{combine_subject}' 的 CSI 文件。"); return
            st.info(f"已拼接 {raw.shape[0]} 时间步 (~{raw.shape[0]/100:.0f}s) 来自用户 '{combine_subject}'。")
        else:
            raw = _load_upload(uploaded, source)
        if use_slice and source == "rssi":
            slices = _slice_rssi(raw, slice_duration_s=slice_duration)
            st.info(f"Sliced into {len(slices)} segments ({slice_duration}s each). Authenticating...")
            slice_results, all_scores_flat = [], []
            for i, seg in enumerate(slices):
                windows = _build_windows(seg)
                if windows.shape[0] == 0: continue
                if mtype == "CNN":
                    _, mean_s, scores = cnn_model.predict_authentication(windows, claimed)
                    threshold = cnn_model.thresholds.get(claimed, 0.5)
                    model_label = "1D-CNN"
                else:
                    feats = _extract_features_for_auth(windows, pca, scaler, feature_config,
                                                       feature_dim=getattr(model, 'feature_dim', None))
                    scores = svm_scores(model.verifiers[claimed], feats)
                    threshold = model.thresholds.get(claimed, 0.5)
                    mean_s = float(np.mean(scores)); model_label = "SVM (RBF)"
                accept = np.mean(scores >= threshold)
                slice_results.append({"segment": i+1, "accept_rate": accept, "mean_score": mean_s,
                                      "decision": mean_s >= threshold, "n_windows": len(scores)})
                all_scores_flat.extend(scores.tolist())
            n_accept = sum(1 for r in slice_results if r["decision"])
            overall_accept_rate = n_accept / max(len(slice_results), 1)
            decision = "接受" if overall_accept_rate >= 0.5 else "拒绝"
            bc = "#E8F5E9" if overall_accept_rate >= 0.5 else "#FFEBEE"
            brc = "#2E7D32" if overall_accept_rate >= 0.5 else "#C62828"
            st.markdown(f"""<div style="padding:20px;border-radius:8px;background:{bc};border:2px solid {brc};text-align:center;margin:10px 0;">
                <h2 style="margin:0;color:{brc};">{decision}</h2>
                <p style="margin:5px 0 0 0;color:#555;font-size:0.95rem;">切片模式 ({slice_duration}s × {len(slice_results)} 段) | 模型: {model_label} | 用户: {claimed} | 通过段数: {n_accept}/{len(slice_results)} ({overall_accept_rate:.1%}) | 阈值: {threshold:.4f}</p></div>""", unsafe_allow_html=True)
            _setup_paper_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            seg_idx = [r["segment"] for r in slice_results]
            seg_rates = [r["accept_rate"] for r in slice_results]
            colors = ['#2E7D32' if r["decision"] else '#C62828' for r in slice_results]
            ax1.bar(seg_idx, seg_rates, color=colors, alpha=0.8, edgecolor='white')
            ax1.axhline(y=0.5, color='#FF6F00', linestyle='--', linewidth=1.5, label='决策边界 (0.5)')
            ax1.set_xlabel("片段编号", fontsize=_FS["large"]); ax1.set_ylabel("接受率 (≥阈值窗口占比)", fontsize=_FS["large"])
            ax1.set_title(f"每段接受率 (阈值={threshold:.4f})", fontsize=_FS["large"])
            ax1.set_ylim(0, 1.05); ax1.legend(fontsize=_FS["normal"]); ax1.grid(axis='y', alpha=0.2)
            ax2.hist(np.array(all_scores_flat), bins=30, color='#4472C4', alpha=0.7, edgecolor='white')
            ax2.axvline(x=threshold, color='#FF6F00', linestyle='--', linewidth=1.5, label=f'阈值={threshold:.4f}')
            ax2.set_xlabel("认证分数", fontsize=_FS["large"]); ax2.set_ylabel("窗口数", fontsize=_FS["large"])
            ax2.set_title(f"分数分布 ({len(all_scores_flat)} 窗口)", fontsize=_FS["large"])
            ax2.legend(fontsize=_FS["normal"])
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("片段通过率", f"{n_accept}/{len(slice_results)} ({overall_accept_rate:.1%})")
            c2.metric("平均分数", f"{float(np.mean(all_scores_flat)):.4f}")
            c3.metric("分数标准差", f"{float(np.std(all_scores_flat)):.4f}")
            c4.metric("总窗口数", str(len(all_scores_flat)))
            c5.metric("平均窗口/段", f"{len(all_scores_flat) / max(len(slice_results), 1):.0f}")
        else:
            windows = _build_windows(raw)
            if windows.shape[0] == 0: st.warning("数据长度不足 (需 ≥ 200 时间点)。"); return
            if mtype == "CNN":
                if claimed not in cnn_model.subjects: st.error(f"用户 '{claimed}' 未注册。"); return
                _, mean_s, scores = cnn_model.predict_authentication(windows, claimed)
                threshold = cnn_model.thresholds.get(claimed, 0.5); model_label = "1D-CNN"
            else:
                if claimed not in model.verifiers: st.error(f"用户 '{claimed}' 未注册。"); return
                feats = _extract_features_for_auth(windows, pca, scaler, feature_config,
                                                   feature_dim=getattr(model, 'feature_dim', None))
                scores = svm_scores(model.verifiers[claimed], feats)
                threshold = model.thresholds.get(claimed, 0.5)
                mean_s = float(np.mean(scores)); model_label = "SVM (RBF)"
            accept_rate = np.mean(scores >= threshold)
            decision = "接受" if mean_s >= threshold else "拒绝"
            bc = "#E8F5E9" if mean_s >= threshold else "#FFEBEE"
            brc = "#2E7D32" if mean_s >= threshold else "#C62828"
            st.markdown(f"""<div style="padding:20px;border-radius:8px;background:{bc};border:2px solid {brc};text-align:center;margin:10px 0;">
                <h2 style="margin:0;color:{brc};">{decision}</h2>
                <p style="margin:5px 0 0 0;color:#555;font-size:0.95rem;">模型: {model_label} | 用户: {claimed} | 平均分数: {mean_s:.4f} | 阈值: {threshold:.4f} | 接受率: {accept_rate:.1%}</p></div>""", unsafe_allow_html=True)
            _score_chart(scores, threshold, f"认证 — {claimed} ({model_label})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("接受率", f"{accept_rate:.1%}"); c2.metric("平均分数", f"{mean_s:.4f}")
            c3.metric("分数标准差", f"{np.std(scores):.4f}"); c4.metric("窗口数", str(len(scores)))
    except Exception as e:
        st.error(f"认证失败: {e}")


def _render_continuous_auth(source, model, pca, scaler, cnn_model, feature_config=None):
    st.subheader("持续认证监控")
    st.caption("实时滑动窗口身份验证。")
    use_combine_cont = False
    if source == "csi":
        na = len(_CONFIG.csi_selected_actions) if _CONFIG.csi_selected_actions else 55
        use_combine_cont = st.checkbox(f"组合模式 ({na} 动作 → ~{na*5}s, 可比肩 RSSI 200s)", False, key="csi_combine_cont")
        if use_combine_cont:
            c_s, c_t = st.columns(2)
            with c_s: combine_subject_cont = st.selectbox("用户", _get_available_subjects(model, cnn_model, source), key="csi_comb_subj_cont")
            with c_t: combine_trial_cont = st.number_input("试验编号", 1, 20, 1, key="csi_comb_trial_cont")
    if not use_combine_cont:
        uploaded = st.file_uploader("上传样本文件", type=["mat"] if source == "rssi" else ["npy"], key="cont_auth")
        if not uploaded: st.info("上传样本文件以开始持续监控。"); return
    c1, c2, c3 = st.columns(3)
    with c1: mtype = st.radio("模型", ["SVM", "CNN"], horizontal=True, key="mt_cont")
    with c2: claimed = st.selectbox("声明身份", _get_available_subjects(model, cnn_model, source), key="claim_cont")
    with c3: ws = st.slider("平滑窗口", 3, 30, 10, key="ws_cont")
    if (mtype == "SVM" and model is None) or (mtype == "CNN" and cnn_model is None):
        st.warning(f"{mtype} 模型未训练。"); return
    if not st.button("开始监控", type="primary", key="btn_cont"): return
    try:
        if use_combine_cont and source == "csi":
            raw = _combine_csi_files(_CONFIG.npy_dir, combine_subject_cont, combine_trial_cont)
            if raw is None: st.error(f"未找到用户 '{combine_subject_cont}' 的 CSI 文件。"); return
            st.info(f"已拼接 {raw.shape[0]} 时间步 (~{raw.shape[0]/100:.0f}s) 来自用户 '{combine_subject_cont}'。")
        else:
            raw = _load_upload(uploaded, source)
        windows = _build_windows(raw)
        if windows.shape[0] == 0: st.warning("数据长度不足。"); return
        if mtype == "CNN":
            if claimed not in cnn_model.subjects: st.error(f"用户 '{claimed}' 未注册。"); return
            threshold = cnn_model.thresholds.get(claimed, 0.5)
            all_scores = cnn_model.predict_authentication(windows, claimed)[2]
            model_label = "1D-CNN"
        else:
            if claimed not in model.verifiers: st.error(f"用户 '{claimed}' 未注册。"); return
            feats = _extract_features_for_auth(windows, pca, scaler, feature_config,
                                               feature_dim=getattr(model, 'feature_dim', None))
            all_scores = svm_scores(model.verifiers[claimed], feats)
            threshold = model.thresholds.get(claimed, 0.5); model_label = "SVM (RBF)"
        n = len(all_scores)
        smoothed = np.array([np.mean(all_scores[max(0, i - ws + 1):i + 1]) for i in range(n)])
        decisions = smoothed >= threshold
        _setup_paper_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [4, 1]})
        t = np.arange(n)
        ax1.fill_between(t, threshold, 1.0, alpha=0.08, color='#2E7D32')
        ax1.fill_between(t, 0, threshold, alpha=0.08, color='#C62828')
        ax1.plot(t, all_scores, alpha=0.25, color='#9E9E9E', lw=0.5, label='原始分数 (逐窗口)')
        ax1.plot(t, smoothed, color='#1565C0', lw=1.8, label=f'平滑分数 (窗口={ws})')
        ax1.axhline(y=threshold, color='#FF6F00', linestyle='--', lw=1.8, label=f'阈值 = {threshold:.4f}')
        ax1.set_ylabel("认证分数 (0~1, 越高越可信)", fontsize=_FS["large"])
        ax1.set_title(f"持续认证 — 用户 {claimed} ({model_label})", fontsize=_FS["large"])
        ax1.legend(loc='upper right', fontsize=_FS["normal"], framealpha=0.9)
        ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.25)
        colors = ['#2E7D32' if d else '#C62828' for d in decisions]
        ax2.bar(t, np.ones(n), width=1.0, color=colors, alpha=0.7)
        ax2.set_xlabel("窗口序号", fontsize=_FS["large"]); ax2.set_ylabel("决策", fontsize=_FS["large"])
        ax2.set_yticks([]); ax2.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        longest = streak = 0
        for d in decisions:
            if d: streak += 1; longest = max(longest, streak)
            else: streak = 0
        switches = sum(1 for i in range(1, n) if decisions[i] != decisions[i-1])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("接受率", f"{np.mean(decisions):.1%}"); c2.metric("最终决策", "接受" if decisions[-1] else "拒绝")
        c3.metric("最长连续接受", f"{longest} 窗口"); c4.metric("状态切换次数", str(switches)); c5.metric("总窗口数", str(n))
        if switches > 2: st.warning(f"检测到 {switches} 次状态切换 — 可能存在身份变化。")
    except Exception as e:
        st.error(f"持续认证失败: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── 实验面板渲染
# ══════════════════════════════════════════════════════════════════════════════

def _render_experiments(source):
    st.subheader("🧪 实验运行面板")
    st.caption("一键运行论文实验 (E1~E5)，结果以图表与表格呈现。")
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
    with c1: st.number_input("随机种子", 0, 999, 42, key="exp_seed")
    with c2:
        exp_select = st.multiselect("选择实验", exp_options, default=exp_defaults[:2],
                                     key="exp_select", help=f"当前数据源: {source.upper()}。仅显示兼容的实验。")
    incompatible = [e for e in exp_select if source not in all_exps[e]["sources"]]
    if incompatible: st.warning(f"以下实验与当前数据源 ({source.upper()}) 不兼容, 将被跳过: {', '.join(incompatible)}")
    op_ui = TrainingOperationUI(start_key="exp_start", cancel_key="exp_cancel")
    def _on_exp_start():
        if not exp_select: st.warning("请至少选择一个实验"); st.stop()
        valid = [e for e in exp_select if source in all_exps[e]["sources"]]
        if not valid: st.error(f"所选实验均与当前数据源 ({source.upper()}) 不兼容, 无法运行。"); st.stop()
        SessionStateManager.reset_training()
    op_ui.render_controls(start_label="▶️ 运行实验", on_start=_on_exp_start)
    is_running = SessionStateManager.is_running()
    has_result = st.session_state.get("exp_results") is not None
    if is_running or has_result:
        if is_running and not SessionStateManager.is_launched() and not has_result:
            _launch_training(lambda: _ExperimentWrapper(source, exp_select), "exp_results", op_ui)
            is_running = SessionStateManager.is_running()
            has_result = st.session_state.get("exp_results") is not None
        if has_result or st.session_state.training_error:
            op_ui.render_results("exp_results")
        if has_result:
            from experiments import (render_e1_results, render_e2_results,
                                     render_e3_results, render_e5_results)
            res = st.session_state.exp_results
            if "E1" in res: render_e1_results(res["E1"])
            if "E2" in res: render_e2_results(res["E2"])
            if "E3" in res: render_e3_results(res["E3"])
            if "E4" in res: _render_compare_results(res["E4"], "")
            if "E5" in res: render_e5_results(res["E5"])


# ══════════════════════════════════════════════════════════════════════════════
# ── 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SessionStateManager.init()
    st.markdown('<div class="auth-header">🔐 身份认证系统</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 阶段")
        phase = st.radio("选择阶段", ["📝 注册阶段 (训练)", "🔍 认证阶段 (推理)"], key="phase")
        st.markdown("---"); st.markdown("## 数据源")
        source_label = st.radio("选择数据源", ["📡 静态 (RSSI/MAT)", "📶 动态 (CSI/NPY)"], key="source")
        st.markdown("---"); st.markdown("## 系统信息")
        st.markdown(f'<div class="memory-info">系统内存: {_memory_monitor.system_memory_pressure:.0%}<br>GPU内存: {_memory_monitor.gpu_memory_pressure:.0%}<br>项目: <code>{_CONFIG.root_dir}</code><br></div>', unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("💾 缓存管理", expanded=False):
            cache = st.session_state.pipeline_cache
            if cache: st.write(f"缓存项: {len(cache._cache_index)}")
            if st.button("🗑️ 清空所有缓存", key="clear_cache"):
                cache.clear(); st.success("缓存已清空"); st.rerun()
    source: DataSource = "rssi" if "静态" in source_label else "csi"
    is_reg = "注册" in phase
    st.markdown(f'<div class="phase-label">{"📝 注册阶段 — 模型训练" if is_reg else "🔍 认证阶段 — 模型推理"}</div>', unsafe_allow_html=True)
    if is_reg:
        ensure_server()
        log_html = write_log_html(_CONFIG.cache_dir)
        st.iframe(src=str(log_html), height=340)
        tab1, tab2, tab3, tab4 = st.tabs(["🔧 基础训练", "📈 参数研究", "⚖️ 模型对比", "🧪 实验"])
        with tab1: _render_register_basic(source)
        with tab2: _render_param_study(source)
        with tab3: _render_model_compare(source)
        with tab4: _render_experiments(source)
    else:
        model, pca, scaler = _load_auth_model(source)
        cnn_model = _load_cnn_model(source)
        if model: _normalize_model_keys(model, source)
        if cnn_model: _normalize_model_keys(cnn_model, source)
        fc = getattr(model, 'feature_config', None)
        st.info(f"已加载模型: {'SVM ✅' if model else 'SVM ⚠️ 未训练'} | {'CNN ✅' if cnn_model else 'CNN ⚠️ 未训练'}。请选择认证模式。")
        tab1, tab2 = st.tabs(["🔑 单次认证", "📊 持续认证"])
        with tab1: _render_single_auth(source, model, pca, scaler, cnn_model, fc)
        with tab2: _render_continuous_auth(source, model, pca, scaler, cnn_model, fc)


if __name__ == "__main__":
    main()
