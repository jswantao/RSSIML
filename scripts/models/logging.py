# -*- coding: utf-8 -*-
"""训练日志记录 — 结构化训练事件日志。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_training(
    model_type: str,
    task_type: str,
    data_source: str,
    status: str,
    duration: float = 0.0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
    pipeline_config: Any = None,
) -> None:
    """记录训练事件日志 (结构化 JSON)。

    Args:
        model_type: 模型类型 ("SVM" / "CNN")。
        task_type: 任务类型 ("authentication")。
        data_source: 数据源 ("rssi" / "csi")。
        status: 状态 ("success" / "failed")。
        duration: 训练耗时 (秒)。
        config: 训练配置字典。
        metrics: 评估指标字典。
        error: 错误信息 (仅 status="failed")。
        pipeline_config: PipelineConfig 实例。
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "task_type": task_type,
        "data_source": data_source,
        "status": status,
        "duration_s": round(duration, 2),
    }
    if config:
        entry["config"] = config
    if metrics:
        # 移除大型嵌套结构, 只保留顶层指标
        entry["metrics"] = {
            k: v for k, v in metrics.items()
            if k != "user_metrics" and not isinstance(v, (dict, list))
        }
    if error:
        entry["error"] = error

    # 日志输出
    if status == "success":
        logger.info(
            "训练完成: %s %s (%s) — %.1fs",
            model_type, task_type, data_source, duration,
        )
    else:
        logger.error(
            "训练失败: %s %s (%s) — %s",
            model_type, task_type, data_source, error,
        )

    # 持久化到日志文件
    if pipeline_config is not None:
        try:
            log_dir = pipeline_config.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "training_history.jsonl"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("训练日志持久化失败: %s", e)
