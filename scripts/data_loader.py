# -*- coding: utf-8 -*-
"""统一数据加载模块 — 同时支持 MAT (RSSI) 和 NPY (CSI) 格式。

数据组织:
  MAT (raw/):  5 用户 × 4 会话 = 20 个文件, 命名 wipin_<subject><session>.mat
  NPY (WiFi/): 19 用户 × 55 动作 × 20 重复 = 20900 个文件, 命名 {subject}_{activity}_{trial}.npy

优化记录 (v2.2):
- 🔴 彻底移除全局单例 (_global_data_cache / _global_memory_pool / _GlobalContextAdapter)
- 🟠 强化 DataLoadContext 为唯一上下文载体，强制显式生命周期管理
- 🟠 DataCache 双重检查锁优化，避免高并发重复加载
- 🟢 全面采用 Python 3.10+ 类型注解与现代化标准
- 🟢 日志命名空间修正为 __name__，避免全局污染
"""
from __future__ import annotations

import logging
import re
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
from scipy.io import loadmat

logger = logging.getLogger(__name__)

__all__ = [
    "DataFormatError",
    "DataCache",
    "DataLoadContext",
    "MemoryPool",
    "AsyncBatchLoader",
    "load_rssi_data",
    "load_rssi_data_generator",
    "load_csi_data",
    "load_csi_data_generator",
    "validate_matrix",
    "parse_npy_filename",
    "load_npy_matrix",
    "estimate_matrix_memory",
]

_NAN_MAX_RATIO = 0.1
_DEFAULT_BATCH_SIZE = 4
_MAX_CACHE_SIZE = 128
_MEMORY_PRESSURE_THRESHOLD = 0.85


# ══════════════════════════════════════════════════════════════════════
# 通用异常
# ══════════════════════════════════════════════════════════════════════
class DataFormatError(RuntimeError):
    """数据格式异常 — 携带 file_path 上下文信息。"""

    def __init__(self, message: str, file_path: Path | None = None):
        self.file_path = file_path
        prefix = f"[{file_path.name}] " if file_path else ""
        super().__init__(f"{prefix}{message}")


# ══════════════════════════════════════════════════════════════════════
# 内存池管理
# ══════════════════════════════════════════════════════════════════════
class MemoryPool:
    """预分配内存池 — 减少大规模矩阵处理中的动态分配开销。
    
    Attributes:
        max_pool_size: 最大池化矩阵数
        hit_rate: 缓存命中率统计
    """

    def __init__(self, max_pool_size: int = 16):
        self.max_pool_size = max_pool_size
        self._pools: dict[tuple[int, ...], list[np.ndarray]] = {}
        self._locks: dict[tuple[int, ...], threading.Lock] = {}
        self.hits = 0
        self.misses = 0

    def acquire(self, shape: tuple[int, ...], dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
        """获取或创建一个指定形状的矩阵。"""
        if shape not in self._locks:
            self._locks[shape] = threading.Lock()
            self._pools[shape] = []

        with self._locks[shape]:
            if self._pools[shape]:
                self.hits += 1
                return self._pools[shape].pop()
            self.misses += 1
            return np.empty(shape, dtype=dtype)

    def release(self, matrix: np.ndarray) -> None:
        """归还矩阵到池中供复用。"""
        shape = matrix.shape
        if shape not in self._pools:
            self._pools[shape] = []
            self._locks[shape] = threading.Lock()

        with self._locks[shape]:
            if len(self._pools[shape]) < self.max_pool_size:
                self._pools[shape].append(matrix)

    @property
    def hit_rate(self) -> float:
        """返回缓存命中率。"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """清空所有池，释放内存。"""
        for shape in self._pools:
            self._pools[shape].clear()
        self.hits = 0
        self.misses = 0
        logger.debug("内存池已清空")

    def __del__(self) -> None:
        self.clear()


# ══════════════════════════════════════════════════════════════════════
# 数据缓存管理
# ══════════════════════════════════════════════════════════════════════
def _estimate_bytes(data: Any) -> int:
    """估算缓存项的内存占用 (bytes)。"""
    if hasattr(data, "nbytes"):
        return data.nbytes
    if isinstance(data, tuple):
        return sum(getattr(d, "nbytes", 0) for d in data)
    return 0


class DataCache:
    """线程安全的 LRU 数据缓存 — 避免重复加载同一文件。
    
    Attributes:
        max_size: 最大缓存项数
        current_size: 当前缓存项数
    """

    def __init__(self, max_size: int = _MAX_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._total_memory_bytes = 0
        self._max_memory_bytes = 2 * 1024 * 1024 * 1024  # 2GB 限制
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """获取缓存或调用 loader 加载新数据。"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                logger.debug("缓存命中: %s", key)
                return self._cache[key]

        self.misses += 1
        data = loader()

        with self._lock:
            # 双重检查：防止并发插入重复键
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            self._cache[key] = data
            self._cache.move_to_end(key)
            data_bytes = _estimate_bytes(data)

            # 驱逐最旧条目
            while len(self._cache) > self.max_size or (self._total_memory_bytes + data_bytes) > self._max_memory_bytes:
                old_key, old_data = self._cache.popitem(last=False)
                self._total_memory_bytes -= _estimate_bytes(old_data)
                logger.debug("缓存驱逐: %s", old_key)

            self._total_memory_bytes += data_bytes

        return data

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._cache.clear()
            self._total_memory_bytes = 0
        logger.debug("数据缓存已清空")

    @property
    def current_size(self) -> int:
        return len(self._cache)

    @property
    def memory_usage_mb(self) -> float:
        return self._total_memory_bytes / (1024 * 1024)


# ══════════════════════════════════════════════════════════════════════
# 数据加载上下文
# ══════════════════════════════════════════════════════════════════════
class DataLoadContext:
    """数据加载上下文 — 聚合缓存和内存池，支持独立实例。
    每个流水线入口应创建自己的上下文实例，避免跨任务数据污染。
    支持上下文管理器协议，退出时自动清理。
    """

    def __init__(self, cache_max_size: int = _MAX_CACHE_SIZE):
        self.data_cache = DataCache(max_size=cache_max_size)
        self.memory_pool = MemoryPool()

    def __enter__(self) -> DataLoadContext:
        return self

    def __exit__(self, *args: Any) -> None:
        self.clear()

    def clear(self) -> None:
        self.data_cache.clear()
        self.memory_pool.clear()

    @property
    def cache_stats(self) -> dict[str, Any]:
        return {
            "cache_size": self.data_cache.current_size,
            "cache_memory_mb": round(self.data_cache.memory_usage_mb, 2),
            "pool_hit_rate": round(self.memory_pool.hit_rate, 4),
        }


# ══════════════════════════════════════════════════════════════════════
# 异步批量加载器
# ══════════════════════════════════════════════════════════════════════
class AsyncBatchLoader:
    """异步批量加载器 — 使用线程池并行加载文件。
    适用于 I/O 密集型的大规模 NPY 文件加载（20000+ 文件）。
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def load_batches(
        self,
        file_paths: list[Path],
        batch_size: int = 32,
        loader_fn: Callable[[Path], tuple[str, int, np.ndarray]] | None = None,
        validate: bool = True,
    ) -> Iterator[list[tuple[str, int, np.ndarray]]]:
        """批量异步加载文件。"""
        if loader_fn is None:
            loader_fn = self._default_npy_loader

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i : i + batch_size]
                futures = {
                    executor.submit(self._safe_load, fp, loader_fn, validate): fp
                    for fp in batch
                }

                batch_results: list[tuple[str, int, np.ndarray]] = []
                failed_count = 0

                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        result = future.result(timeout=30)
                        if result is not None:
                            batch_results.append(result)
                    except Exception as e:
                        failed_count += 1
                        logger.error("异步加载失败: %s - %s", fp.name, e)

                if failed_count > 0:
                    logger.warning("批次中 %d 个文件加载失败", failed_count)

                yield batch_results

    @staticmethod
    def _default_npy_loader(file_path: Path) -> tuple[str, int, np.ndarray]:
        subject, session = parse_npy_filename(file_path)
        matrix = load_npy_matrix(file_path)
        return subject, session, matrix

    @staticmethod
    def _safe_load(
        file_path: Path,
        loader_fn: Callable,
        validate: bool,
    ) -> tuple[str, int, np.ndarray] | None:
        try:
            subject, session, matrix = loader_fn(file_path)
            if validate:
                warnings_list = _validate_matrix(matrix, file_path.name)
                for w in warnings_list:
                    logger.warning("数据质量警告 - %s: %s", file_path.name, w)
            return subject, session, matrix
        except Exception as e:
            logger.error("加载失败: %s - %s", file_path.name, e)
            return None


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def estimate_matrix_memory(shape: tuple[int, ...], dtype: np.dtype = np.dtype("float32")) -> int:
    """估算矩阵内存占用（字节）。"""
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def _check_memory_pressure() -> bool:
    """检测系统内存压力。"""
    try:
        mem = psutil.virtual_memory()
        return mem.percent > (_MEMORY_PRESSURE_THRESHOLD * 100)
    except Exception:
        return False


def _collect_files(directory: Path, pattern: str, recursive: bool) -> list[Path]:
    """收集符合模式的文件路径。"""
    if not directory.exists():
        raise FileNotFoundError(f"数据目录不存在: {directory}")
    files = sorted(directory.rglob(pattern) if recursive else directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"目录中未找到 '{pattern}' 文件: {directory}")
    return files


# ══════════════════════════════════════════════════════════════════════
# MAT (RSSI) — 5 用户 × 4 会话
# ══════════════════════════════════════════════════════════════════════
MAT_PATTERN = re.compile(r"^wipin([A-Za-z0-9]+)(\d+).mat$")


def load_rssi_data(
    data_dir: Path,
    recursive: bool = False,
    validate: bool = True,
    use_cache: bool = True,
    context: DataLoadContext | None = None,
) -> list[tuple[str, int, np.ndarray]]:
    """加载 raw/ 目录下所有 wipin_<subject><session>.mat 文件。"""
    results = list(
        load_rssi_data_generator(
            data_dir, recursive, validate, use_cache=use_cache, context=context
        )
    )
    if not results:
        raise DataFormatError("所有文件加载均失败", file_path=data_dir)
    return results


def load_rssi_data_generator(
    data_dir: Path,
    recursive: bool = False,
    validate: bool = True,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
    use_async: bool = False,
    max_workers: int = 4,
    context: DataLoadContext | None = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """流式加载 RSSI 数据 — 分批处理降低内存峰值。"""
    if context is None:
        warnings.warn("未提供 DataLoadContext，将创建临时上下文。建议显式传递以控制生命周期。", UserWarning)
        context = DataLoadContext()

    mat_files = _collect_files(data_dir, "*.mat", recursive)
    logger.info("找到 %d 个 MAT 文件（分批大小: %d）", len(mat_files), batch_size)

    if _check_memory_pressure():
        logger.warning("检测到高内存压力，强制使用小批次")
        batch_size = min(batch_size, 2)
        use_async = False

    failed_total = 0
    success_total = 0

    if use_async and len(mat_files) > batch_size:
        loader = AsyncBatchLoader(max_workers=max_workers)
        for batch in loader.load_batches(
            mat_files,
            batch_size,
            loader_fn=lambda fp: _load_mat_with_cache(fp, validate, use_cache, context),
            validate=False,
        ):
            success_total += len(batch)
            yield from batch
    else:
        for i in range(0, len(mat_files), batch_size):
            batch = mat_files[i : i + batch_size]
            batch_results, batch_failed = _process_mat_batch(batch, validate, use_cache, context)
            success_total += len(batch_results)
            failed_total += batch_failed
            yield from batch_results

    if failed_total:
        logger.warning("MAT 加载: %d 成功, %d 失败", success_total, failed_total)
    if success_total == 0:
        raise DataFormatError(f"所有 {failed_total} 个 MAT 文件加载均失败", file_path=data_dir)


def _process_mat_batch(
    file_batch: list[Path],
    validate: bool,
    use_cache: bool = True,
    context: DataLoadContext | None = None,
) -> tuple[list[tuple[str, int, np.ndarray]], int]:
    results: list[tuple[str, int, np.ndarray]] = []
    failed = 0
    for fp in file_batch:
        try:
            result = _load_mat_with_cache(fp, validate, use_cache, context)
            if result is not None:
                results.append(result)
        except DataFormatError as e:
            failed += 1
            logger.error("加载失败: %s - %s", fp.name, e)
        except Exception as e:
            failed += 1
            logger.exception("未知错误: %s", fp.name)
    return results, failed


def _load_mat_with_cache(
    file_path: Path,
    validate: bool,
    use_cache: bool,
    context: DataLoadContext | None = None,
) -> tuple[str, int, np.ndarray] | None:
    if context is None:
        return None
    cache = context.data_cache
    cache_key = f"mat:{file_path}"
    try:
        if use_cache:
            return cache.get_or_load(cache_key, lambda: _load_mat_uncached(file_path, validate))
        return _load_mat_uncached(file_path, validate)
    except Exception:
        return None


def _load_mat_uncached(file_path: Path, validate: bool) -> tuple[str, int, np.ndarray]:
    subject, session = _parse_mat_filename(file_path)
    matrix = _load_mat_matrix(file_path)
    if validate:
        for w in _validate_matrix(matrix, file_path.name):
            logger.warning("数据质量警告 - %s: %s", file_path.name, w)
    return subject, session, matrix


def validate_matrix(matrix: np.ndarray, file_name: str = "") -> list[str]:
    """验证数据矩阵质量（公开接口）。"""
    return _validate_matrix(matrix, file_name)


def _parse_mat_filename(file_path: Path) -> tuple[str, int]:
    m = MAT_PATTERN.match(file_path.name)
    if not m:
        raise DataFormatError(
            f"文件名格式异常: {file_path.name}, 预期 wipin<subject><session>.mat",
            file_path=file_path,
        )
    return m.group(1), int(m.group(2))


def _load_mat_matrix(file_path: Path) -> np.ndarray:
    try:
        data = loadmat(file_path, squeeze_me=True, simplify_cells=True)
    except Exception as e:
        raise DataFormatError(f"MAT 加载失败: {file_path.name}", file_path=file_path) from e

    if "RSSI" not in data:
        available = [k for k in data if not k.startswith(("__", "_"))]
        raise DataFormatError(f"缺少 'RSSI' 键, 可用: {available}", file_path=file_path)

    matrix = np.asarray(data["RSSI"])
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32, copy=False)
    if matrix.ndim != 2:
        raise DataFormatError(f"维度异常: {matrix.shape}", file_path=file_path)
    if matrix.size == 0:
        raise DataFormatError("矩阵为空", file_path=file_path)
    return matrix


# ══════════════════════════════════════════════════════════════════════
# NPY (CSI) — 19 用户 × 55 动作 × 20 重复
# ══════════════════════════════════════════════════════════════════════
NPY_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+).npy$")


def load_csi_data(
    data_dir: Path,
    recursive: bool = False,
    validate: bool = True,
    use_cache: bool = True,
    use_async: bool = False,
    max_workers: int = 8,
    context: DataLoadContext | None = None,
) -> list[tuple[str, int, np.ndarray]]:
    """加载 WiFi/ 目录下所有 {subject}_{activity}_{trial}.npy 文件。"""
    results = list(
        load_csi_data_generator(
            data_dir, recursive, validate, use_cache=use_cache, use_async=use_async, max_workers=max_workers, context=context
        )
    )
    if not results:
        raise DataFormatError("所有 NPY 文件加载均失败", file_path=data_dir)
    return results


def load_csi_data_generator(
    data_dir: Path,
    recursive: bool = False,
    validate: bool = True,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
    use_async: bool = False,
    max_workers: int = 8,
    context: DataLoadContext | None = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """流式加载 CSI 数据 — 分批处理降低内存峰值。"""
    if context is None:
        warnings.warn("未提供 DataLoadContext，将创建临时上下文。建议显式传递以控制生命周期。", UserWarning)
        context = DataLoadContext()

    npy_files = _collect_files(data_dir, "*.npy", recursive)
    logger.info("找到 %d 个 NPY 文件（分批大小: %d）", len(npy_files), batch_size)

    if _check_memory_pressure():
        logger.warning("检测到高内存压力，自动禁用缓存并减小批次")
        batch_size = min(batch_size, 2)
        use_cache = False
        use_async = False

    failed_total = 0
    success_total = 0

    if use_async and len(npy_files) > batch_size:
        loader = AsyncBatchLoader(max_workers=max_workers)
        for batch in loader.load_batches(
            npy_files,
            batch_size,
            loader_fn=lambda fp: _load_npy_with_cache(fp, validate, use_cache, context),
            validate=False,
        ):
            success_total += len(batch)
            yield from batch
    else:
        for i in range(0, len(npy_files), batch_size):
            batch = npy_files[i : i + batch_size]
            batch_results, batch_failed = _process_npy_batch(batch, validate, use_cache, context)
            success_total += len(batch_results)
            failed_total += batch_failed
            yield from batch_results

    if failed_total:
        logger.warning("NPY 加载: %d 成功, %d 失败", success_total, failed_total)
    if success_total == 0:
        raise DataFormatError(f"所有 {failed_total} 个 NPY 文件加载均失败", file_path=data_dir)


def _process_npy_batch(
    file_batch: list[Path],
    validate: bool,
    use_cache: bool = True,
    context: DataLoadContext | None = None,
) -> tuple[list[tuple[str, int, np.ndarray]], int]:
    results: list[tuple[str, int, np.ndarray]] = []
    failed = 0
    for fp in file_batch:
        try:
            result = _load_npy_with_cache(fp, validate, use_cache, context)
            if result is not None:
                results.append(result)
        except (ValueError, OSError) as e:
            failed += 1
            logger.error("加载失败: %s - %s", fp.name, e)
        except Exception as e:
            failed += 1
            logger.exception("未知错误: %s", fp.name)
    return results, failed


def _load_npy_with_cache(
    file_path: Path,
    validate: bool,
    use_cache: bool,
    context: DataLoadContext | None = None,
) -> tuple[str, int, np.ndarray] | None:
    if context is None:
        return None
    cache = context.data_cache
    cache_key = f"npy:{file_path}"
    try:
        if use_cache:
            return cache.get_or_load(cache_key, lambda: _load_npy_uncached(file_path, validate))
        return _load_npy_uncached(file_path, validate)
    except Exception:
        return None


def _load_npy_uncached(file_path: Path, validate: bool) -> tuple[str, int, np.ndarray]:
    subject, session = parse_npy_filename(file_path)
    matrix = load_npy_matrix(file_path)
    if validate:
        for w in _validate_matrix(matrix, file_path.name):
            logger.warning("数据质量警告 - %s: %s", file_path.name, w)
    return subject, session, matrix


def parse_npy_filename(file_path: Path) -> tuple[str, int]:
    """解析 {subject}_{activity}_{trial}.npy, 返回 (subject, session)。"""
    m = NPY_PATTERN.match(file_path.name)
    if not m:
        raise ValueError(
            f"文件名格式异常: {file_path.name}, 预期 <subject>_<activity>_<trial>.npy"
        )
    return m.group(1), int(m.group(2)) * 100 + int(m.group(3))


def load_npy_matrix(file_path: Path) -> np.ndarray:
    """加载 NPY 并转置 (features, time) → (time, features)。"""
    try:
        data = np.load(file_path)
    except (ValueError, OSError) as e:
        raise ValueError(f"NPY 加载失败: {file_path.name}") from e
    if data.ndim != 2:
        raise ValueError(f"维度异常: {data.shape}")
    if data.size == 0:
        raise ValueError(f"矩阵为空: {file_path.name}")
    data = np.asarray(data.T)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return data


# ══════════════════════════════════════════════════════════════════════
# 内部通用 — 数据验证
# ══════════════════════════════════════════════════════════════════════
def _validate_matrix(matrix: np.ndarray, file_name: str = "") -> list[str]:
    """统一的数据质量验证（向量化优化）。"""
    warnings_list: list[str] = []
    if matrix.size == 0:
        warnings_list.append("矩阵为空")
        return warnings_list

    nan_mask = np.isnan(matrix)
    nan_ratio = float(np.mean(nan_mask))
    if nan_ratio > _NAN_MAX_RATIO:
        warnings_list.append(f"NaN 比例过高: {nan_ratio:.2%}")

    inf_ratio = float(np.mean(np.isinf(matrix)))
    if inf_ratio > 0:
        warnings_list.append(f"包含 Inf: {inf_ratio:.2%}")

    if matrix.shape[1] > 0 and matrix.size > 0:
        valid_mask = ~nan_mask
        has_valid = np.any(valid_mask, axis=0)
        col_std = np.nanstd(matrix, axis=0)
        constant_cols = np.where(has_valid & (col_std < 1e-8))[0]

        if len(constant_cols) > 0:
            first_const_col = constant_cols[0]
            first_valid_idx = np.where(valid_mask[:, first_const_col])[0][0]
            example_val = matrix[first_valid_idx, first_const_col]
            warnings_list.append(
                f"第 {first_const_col} 列为常数值 {example_val} (共 {len(constant_cols)} 个常数列)"
            )

    return warnings_list