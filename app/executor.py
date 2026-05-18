# -*- coding: utf-8 -*-
"""训练执行器 — TrainingExecutor 基类 + 具体实现 (基础/参数研究/模型对比/实验)。"""
from __future__ import annotations

import gc
import time
import traceback
from abc import ABC, abstractmethod

import numpy as np
import streamlit as st

from scripts.models import clear_gpu_memory, log_training
from scripts.pipeline_runner import AuthPipeline

from app.state import SessionStateManager, _CONFIG, app_logger
from app.ui import TrainingCancelledError, check_cancelled, make_progress_callback


class TrainingExecutor(ABC):
    """抽象训练执行器 — 统一生命周期 (启动→执行→完成/取消/错误)。"""

    def __init__(self, mtype: str, source: str):
        self.mtype = mtype
        self.source = source
        self.logger = app_logger
        self._op_ui = None

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
        check_cancelled()
        result = self._execute_core()
        self._on_success(result)
        return result

    def _on_start(self):
        self._update_ui(0.05, "初始化流水线...", "training-running", "🔄 训练进行中...")
        self.logger.info(f"开始 {self.mtype} 训练")

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
    def _execute_core(self): ...

    def get_config(self) -> dict:
        return {}


class BasicTrainingExecutor(TrainingExecutor):
    """基础注册训练 — SVM 或 CNN。"""

    def __init__(self, source, mtype, pipeline_kwargs, extra_kwargs):
        super().__init__(mtype, source)
        self.pipeline_kwargs = pipeline_kwargs
        self.extra_kwargs = extra_kwargs

    def get_config(self):
        kw = self.pipeline_kwargs
        c = {"data_source": self.source, "seed": kw.get("seed"),
             "test_size": kw.get("test_size"), "window_size": kw.get("window_size"),
             "step_size": kw.get("step_size"), "use_pca": kw.get("use_pca"),
             "threshold_method": kw.get("threshold_method"),
             "max_files_per_subject": kw.get("max_files_per_subject"),
             "use_online_svm": kw.get("use_online_svm", False),
             "feature_groups": list(kw.get("feature_groups", [])),
             "csi_denoise": kw.get("csi_denoise")}
        if kw.get("use_online_svm"):
            c["online_kernel"] = kw.get("online_kernel", "linear")
        c.update(self.extra_kwargs)
        return c

    def _execute_core(self):
        check_cancelled()
        self._update_ui(0.15, "数据划分中...", "training-running", "🔄 训练进行中...")
        kwargs = dict(self.pipeline_kwargs)
        kwargs["progress_callback"] = make_progress_callback(0.15, 0.55, self._op_ui)
        pipeline = AuthPipeline(**kwargs)
        check_cancelled()
        self._update_ui(0.60, f"训练 {self.mtype} 模型中...", "training-running", "🔄 训练进行中...")
        self.logger.info(f"开始 {self.mtype} 模型训练...")
        if self.mtype == "SVM":
            result = pipeline.run_svm()
        else:
            result = pipeline.run_cnn(cancel_fn=check_cancelled, **self.extra_kwargs)
        check_cancelled()
        self._update_ui(0.85, "评估模型性能...", "training-running", "🔄 训练进行中...")
        check_cancelled()
        self._update_ui(0.95, "保存模型...", "training-running", "🔄 训练进行中...")
        return result


class _ExperimentWrapper(TrainingExecutor):
    """实验包装器 — 对接 TrainingExecutor 接口，调用 ExperimentRunner。"""

    def __init__(self, source, exp_select):
        super().__init__("实验", source)
        self.exp_select = exp_select

    def get_config(self):
        return {"data_source": self.source, "experiments": self.exp_select}

    def _execute_core(self):
        from experiments import ExperimentRunner
        runner = ExperimentRunner(self.source, check_cancelled=check_cancelled)
        exp_keys = set()
        skipped = []
        for e in self.exp_select:
            if e.startswith("E1"):
                exp_keys.add("E1")
            elif e.startswith("E2"):
                if self.source == "rssi": exp_keys.add("E2")
                else: skipped.append(e)
            elif e.startswith("E3"):
                if self.source == "csi": exp_keys.add("E3")
                else: skipped.append(e)
            elif e.startswith("E4"):
                exp_keys.add("E4")
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
            check_cancelled()
            progress = (i + 0.1) / max(total, 1)
            self._update_ui(progress, f"运行 {ek}...", "training-running", f"🔄 实验: {ek}")
            if ek == "E1": runner.run_e1()
            elif ek == "E2": runner.run_e2()
            elif ek == "E3": runner.run_e3()
            elif ek == "E4": runner.run_e4()
            elif ek == "E5": runner.run_e5()
            self._update_ui((i + 1) / max(total, 1), f"{ek} 完成", "training-running", f"✅ {ek}")
        st.session_state.exp_results = runner.results
        return runner.results

    def _on_success(self, result):
        self._update_ui(1.0, "训练完成！", "training-completed", "✅ 实验完成！")
