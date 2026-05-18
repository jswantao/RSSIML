# -*- coding: utf-8 -*-
"""训练日志 — JSONL 格式，每次训练成功/失败/取消均记录一条。"""
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def log_training(
    *,
    model_type: str,
    task_type: str,
    data_source: str,
    status: str,  # "success" | "failed" | "cancelled"
    duration: Optional[float] = None,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    error: Optional[str] = None,
    pipeline_config: Optional["PipelineConfig"] = None,
) -> None:
    """记录一条训练日志 (JSONL 追加)。

    Args:
        pipeline_config: 流水线配置实例，用于确定日志输出目录。
                         None 时从 from_root() 推导。
    """
    from scripts.config import PipelineConfig

    entry: dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": model_type,
        "task_type": task_type,
        "data_source": data_source,
        "status": status,
    }
    if duration is not None:
        entry["duration_seconds"] = round(duration, 2)
    if config:
        entry["config"] = _sanitize(config)
    if metrics:
        entry["metrics"] = _sanitize(metrics)
    if error:
        entry["error"] = str(error)[:1000]

    try:
        pcfg = pipeline_config or PipelineConfig.from_root()
        path = pcfg.log_dir / "training_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"无法写入训练日志: {e}")


def _sanitize(d: dict) -> dict:
    """过滤不可 JSON 序列化的值，同时移除 None 值。"""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _sanitize(v)
        else:
            out[k] = str(v)
    return out
