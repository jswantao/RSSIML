# -*- coding: utf-8 -*-
"""内存与 GPU 资源监控 — 训练期间的资源管理。"""
from __future__ import annotations

import gc
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """系统与 GPU 内存监控器。"""

    def __init__(self) -> None:
        self._gpu_available = False
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

    def spawn(self) -> MemoryMonitor:
        """返回自身 (兼容链式调用)。"""
        return self

    def should_degrade(self) -> bool:
        """检测是否应降级 (内存压力高)。"""
        try:
            import psutil
            return psutil.virtual_memory().percent > 85
        except ImportError:
            return False

    def get_optimal_batch_size(self, requested: int) -> int:
        """根据内存状态调整批次大小。"""
        if self.should_degrade():
            return max(16, requested // 2)
        return requested

    def log_status(self) -> None:
        """输出当前内存状态日志。"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            logger.info(
                "内存状态: 已用 %.1f GB / 总计 %.1f GB (%.1f%%)",
                vm.used / (1024**3), vm.total / (1024**3), vm.percent,
            )
        except ImportError:
            logger.debug("psutil 不可用, 跳过内存状态日志")

        if self._gpu_available:
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    logger.info(
                        "GPU %d: 已分配 %.2f GB, 已保留 %.2f GB",
                        i, allocated, reserved,
                    )
            except Exception:
                pass


_monitor_instance: MemoryMonitor | None = None
_monitor_lock = threading.Lock()


def get_memory_monitor() -> MemoryMonitor:
    """获取全局内存监控器 (单例)。"""
    global _monitor_instance
    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = MemoryMonitor()
    return _monitor_instance


def clear_gpu_memory() -> None:
    """释放 GPU 显存。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU 显存已清理")
    except ImportError:
        pass
