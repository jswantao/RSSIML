# -*- coding: utf-8 -*-
"""UI 交互组件 — 操作按钮、进度条、字体、CSS 样式。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

from app.state import SessionStateManager


class TrainingCancelledError(Exception):
    """训练被手动终止的哨兵异常。"""
    pass


def check_cancelled() -> None:
    if SessionStateManager.is_cancelled():
        raise TrainingCancelledError("训练已被手动终止")


# ── 字体 (缓存加载) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_font() -> fm.FontProperties:
    for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/PingFang.ttc"]:
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return fm.FontProperties()


FONT = _get_font()
if FONT:
    plt.rcParams["font.family"] = FONT.get_name()
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False

# ── CSS 样式 ───────────────────────────────────────────────────────────
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


def make_progress_callback(start_fraction: float = 0.15,
                           end_fraction: float = 0.55,
                           op_ui: TrainingOperationUI | None = None):
    span = end_fraction - start_fraction
    def callback(current: int, total: int, message: str = "") -> None:
        fraction = current / max(total, 1)
        progress = start_fraction + fraction * span
        SessionStateManager.update_progress(progress, message)
        if op_ui:
            op_ui.update_progress(progress, message)
    return callback


@dataclass
class TrainingOperationUI:
    """训练操作 UI 组件 — 启动/终止按钮 + 进度条 + 状态显示。"""
    start_key: str
    cancel_key: str
    _status: Any = None
    _progress: Any = None

    def render_controls(self, start_label: str = "🚀 开始训练",
                         on_start: callable | None = None) -> None:
        is_running = SessionStateManager.is_running()
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(start_label, type="primary", key=self.start_key,
                         disabled=is_running):
                if on_start:
                    on_start()
        with col2:
            if is_running:
                if st.button("🛑 终止训练", type="secondary", key=self.cancel_key):
                    SessionStateManager.mark_cancelled()
                    st.rerun()
        self._status = st.empty()
        self._progress = st.empty()

    def update_progress(self, fraction: float, message: str = "") -> None:
        if self._progress:
            self._progress.progress(fraction, text=message)

    def update_status(self, css_class: str, message: str) -> None:
        if self._status:
            self._status.markdown(
                f'<div class="training-status {css_class}"><strong>'
                f'{message}</strong></div>',
                unsafe_allow_html=True,
            )

    def render_results(self, result_key: str = "training_result") -> None:
        has_result = st.session_state.get(result_key) is not None
        has_error = st.session_state.training_error is not None
        if has_result:
            self.update_status("training-completed", "✅ 训练完成！")
        elif has_error:
            self.update_status("training-error",
                             f"❌ 训练失败: {st.session_state.training_error}")
