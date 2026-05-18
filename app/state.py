# -*- coding: utf-8 -*-
"""Session 状态管理 + 全局配置单例。"""
from __future__ import annotations

import logging
from typing import Literal

import streamlit as st

from scripts.config import PipelineConfig
from scripts.models.memory import get_memory_monitor
from scripts.pipeline_runner import PipelineCache
from scripts.log_server import WebSocketLogHandler

DataSource = Literal["rssi", "csi"]

# ── 全局单例 ──────────────────────────────────────────────────────────
_CONFIG = PipelineConfig.from_root()
_CONFIG.ensure_dirs()
_memory_monitor = get_memory_monitor()

st.set_page_config(
    page_title="身份认证系统", page_icon="🔐", layout="wide",
    initial_sidebar_state="expanded",
)

# ── 专用日志器 ────────────────────────────────────────────────────────
app_logger = logging.getLogger("rssi_app")
app_logger.propagate = False
app_logger.setLevel(logging.INFO)
for h in list(app_logger.handlers):
    app_logger.removeHandler(h)
if not any(isinstance(h, WebSocketLogHandler) for h in app_logger.handlers):
    app_logger.addHandler(WebSocketLogHandler())


class SessionStateManager:
    """Streamlit session_state 集中管理器 — 训练生命周期、进度、结果。"""

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
        "pipeline_cache": None,
    }

    @classmethod
    def init(cls) -> None:
        for k, v in cls.DEFAULTS.items():
            if k not in st.session_state:
                st.session_state[k] = v
        if st.session_state.pipeline_cache is None:
            st.session_state.pipeline_cache = PipelineCache(_CONFIG)

    @classmethod
    def reset_training(cls) -> None:
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
    def complete_training(cls) -> None:
        st.session_state.training_active = False

    @classmethod
    def mark_launched(cls) -> None:
        st.session_state.training_launched = True

    @classmethod
    def is_launched(cls) -> bool:
        return st.session_state.get("training_launched", False)

    @classmethod
    def mark_cancelled(cls) -> None:
        st.session_state.training_cancelled = True

    @classmethod
    def is_running(cls) -> bool:
        return st.session_state.training_active

    @classmethod
    def is_cancelled(cls) -> bool:
        return st.session_state.training_cancelled

    @classmethod
    def update_progress(cls, fraction: float, message: str = "") -> None:
        st.session_state.training_progress = fraction
        if message:
            st.session_state.training_message = message

    @classmethod
    def set_error(cls, error: str) -> None:
        st.session_state.training_error = error
