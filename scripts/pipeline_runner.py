# -*- coding: utf-8 -*-
"""统一流水线模块 — RSSI (MAT) 与 CSI (NPY) 通用。

提供身份认证的端到端流水线，支持 SVM 和 1D-CNN，统一 RSSI/CSI 数据源。

优化内容:
  v2.3:
  - 修复参数哈希，确保 max_files_per_subject 参与哈希计算
  - 模型缓存检查前明确判断 use_model_cache 标志
  - 改进日志输出，区分"缓存命中"与"完整训练"
  v2.2:
  - 添加流水线阶段缓存，避免重复计算
  - 内存映射文件自动清理机制
  - 流式处理优化，减少内存峰值
  - 异步数据准备，重叠 I/O 和计算
  - 智能缓存管理，自动清理过期中间文件
  - 流水线执行时间分析和瓶颈检测
  v2.1:
  - 统一数据准备流程
  - 支持断点续传（跳过已完成阶段）
  - 内存使用监控和告警
"""
import gc
import gzip
import json
import logging
import pickle
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Any

import numpy as np

from scripts.build_sliding_windows import WindowBuilder, WindowConfig, WindowProcessor
from scripts.config import PipelineConfig
from scripts.process_features_pca_norm import (
    FeatureExtractor,
    FeatureProcessor,
    PreprocessConfig,
)
from scripts.split_rssi_dataset import DatasetSplitter, _clear_sample_cache
from scripts.data_loader import DataLoadContext
from scripts.models import (
    CNNTrainConfig,
    CNNTrainer,
    get_memory_monitor,
    SVMConfig,
    SVMAuthenticationTrainer,
    clear_gpu_memory,
    log_training,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineParams",
    "PipelineResult",
    "DataPipeline",
    "AuthPipeline",
    "PipelineCache",
    "run_authentication_pipeline",
    "run_data_pipeline",
    "run_npy_authentication_svm",
    "run_npy_authentication_cnn",
]

TaskType = Literal["authentication"]
ModelType = Literal["svm", "cnn"]
DataSource = Literal["rssi", "csi"]

_PARAM_NAMES = (
    "test_size", "random_seed", "window_size",
    "step_size", "pca_variance", "use_pca",
    "feature_groups",
)

_CSI_LOG_INTERVAL_SVM = 50
_CSI_LOG_INTERVAL_CNN = 50
_MAX_CACHE_AGE_DAYS = 7  # 缓存最大保留天数
_MAX_CACHE_SIZE_GB = 50  # 缓存最大总大小


# ══════════════════════════════════════════════════════════════════════════════
# CSI 原始信号降噪 (窗口构建前应用)
# ══════════════════════════════════════════════════════════════════════════════

def _hampel_filter(data: np.ndarray, window: int = 7,
                   n_sigma: float = 3.0) -> np.ndarray:
    """Hampel 滤波器 — 逐时间步检测并替换脉冲异常值。

    CSI 信号常受突发干扰产生脉冲尖峰，Hampel 使用滑动窗口中位数
    和 MAD (Median Absolute Deviation) 检测异常值。

    Args:
        data: (n_timesteps, n_subcarriers) float32。
        window: 滑动窗口大小 (奇数)。
        n_sigma: 异常判定阈值 (默认 3σ)。

    Returns:
        滤波后同形状数组。
    """
    n = data.shape[0]
    half = window // 2
    k = 1.4826  # MAD → 标准差换算系数 (正态分布假设)
    result = data.astype(np.float64, copy=True)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbourhood = result[lo:hi]
        median = np.median(neighbourhood, axis=0)
        mad = np.median(np.abs(neighbourhood - median), axis=0)
        threshold = n_sigma * k * (mad + 1e-10)
        diff = np.abs(result[i] - median)
        mask = diff > threshold
        result[i, mask] = median[mask]
    return result.astype(np.float32)


def _savgol_filter(data: np.ndarray, window: int = 7,
                   order: int = 3) -> np.ndarray:
    """Savitzky-Golay 滤波器 — 保持信号边缘特征的平滑降噪。

    对 CSI 振幅的局部多项式拟合, 比移动平均更好地保留
    信号突变特征 (如动作开始/结束的瞬时变化)。

    Args:
        data: (n_timesteps, n_subcarriers) float32。
        window: 拟合窗口大小 (奇数)。
        order: 多项式阶数。

    Returns:
        平滑后同形状数组。
    """
    from scipy.signal import savgol_filter
    result = savgol_filter(data.astype(np.float64), window, order, axis=0)
    return result.astype(np.float32)


def _butterworth_filter(data: np.ndarray, cutoff: float = 0.4,
                        order: int = 4) -> np.ndarray:
    """巴特沃斯低通滤波器 — 保留低频运动信息, 抑制高频噪声。

    CSI 人体运动信息集中在 30Hz 以下, 采样率 100Hz 时
    cutoff=0.4 → 20Hz 截止, 有效滤除高频硬件噪声。

    Args:
        data: (n_timesteps, n_subcarriers) float32。
        cutoff: 归一化截止频率 (0~1, 相对 Nyquist)。
        order: 滤波器阶数 (越大衰减越陡, 默认 4)。

    Returns:
        滤波后同形状数组。
    """
    from scipy.signal import butter, sosfiltfilt
    if cutoff >= 1.0:
        return data.astype(np.float32)
    sos = butter(order, cutoff, btype='low', output='sos')
    result = sosfiltfilt(sos, data.astype(np.float64), axis=0)
    return result.astype(np.float32)


def _apply_csi_denoise(arr: np.ndarray, method: str | None) -> np.ndarray:
    """根据配置对 CSI 原始信号 (n_timesteps, n_subcarriers) 降噪。"""
    if method is None:
        return arr
    if method == "hampel":
        return _hampel_filter(arr)
    if method == "savgol":
        return _savgol_filter(arr)
    if method == "butterworth":
        return _butterworth_filter(arr)
    return arr


def _current_exception() -> str:
    import sys
    return str(sys.exc_info()[1])


# ══════════════════════════════════════════════════════════════════════════════
# 流水线缓存管理
# ══════════════════════════════════════════════════════════════════════════════

class PipelineCache:
    """流水线阶段缓存 — 避免重复计算，支持断点续传。
    
    缓存策略:
      - 基于文件哈希和参数的 LRU 缓存
      - 自动清理过期缓存（超过 7 天）
      - 内存压力检测，自动降级
      
    Example:
        >>> cache = PipelineCache(PipelineConfig.from_root())
        >>> # 检查是否有缓存的划分结果
        >>> if cache.exists("split", "classification"):
        ...     sr = cache.load("split", "classification")
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index: OrderedDict[str, dict] = OrderedDict()
        self._load_index()
        
    def _load_index(self) -> None:
        """加载缓存索引。"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with index_file.open("r", encoding="utf-8") as f:
                    entries = json.load(f)
                    # 按时间排序
                    sorted_entries = sorted(
                        entries.items(),
                        key=lambda x: x[1].get("timestamp", 0)
                    )
                    self._cache_index = OrderedDict(sorted_entries)
            except Exception:
                self._cache_index = OrderedDict()
        self._cleanup_expired()
    
    def _save_index(self) -> None:
        """保存缓存索引。"""
        index_file = self.cache_dir / "cache_index.json"
        with index_file.open("w", encoding="utf-8") as f:
            json.dump(dict(self._cache_index), f, indent=2)
    
    def exists(self, stage: str, params_hash: str) -> bool:
        """检查缓存是否存在。"""
        key = f"{stage}_{params_hash}"
        if key not in self._cache_index:
            return False
        
        cache_path = Path(self._cache_index[key]["path"])
        return cache_path.exists()
    
    def load(self, stage: str, params_hash: str) -> Any | None:
        """加载缓存的中间结果。"""
        key = f"{stage}_{params_hash}"
        if not self.exists(stage, params_hash):
            return None
        
        cache_path = Path(self._cache_index[key]["path"])
        try:
            with cache_path.open("rb") as f:
                result = pickle.load(f)
            
            # 移到末尾（最近使用）
            self._cache_index.move_to_end(key)
            self._save_index()
            
            logger.info(f"缓存命中: {stage} ({cache_path.name})")
            return result
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None
    
    def save(self, stage: str, params_hash: str, result: Any, 
             compress: bool = False) -> Path:
        """保存中间结果到缓存。"""
        key = f"{stage}_{params_hash}"
        
        # 创建缓存文件
        suffix = ".pkl.gz" if compress else ".pkl"
        cache_file = self.cache_dir / f"{stage}_{params_hash}{suffix}"
        
        # 保存数据
        open_func = gzip.open if compress else open
        with open_func(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 更新索引
        self._cache_index[key] = {
            "stage": stage,
            "params_hash": params_hash,
            "path": str(cache_file),
            "timestamp": time.time(),
            "size_mb": cache_file.stat().st_size / (1024 * 1024),
        }
        
        # 清理旧缓存
        self._cache_index.move_to_end(key)
        self._cleanup_expired()
        self._save_index()
        
        logger.debug(f"缓存已保存: {stage} → {cache_file.name}")
        return cache_file
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存。"""
        current_time = time.time()
        to_remove = []
        total_size_mb = 0
        
        for key, info in self._cache_index.items():
            age_days = (current_time - info.get("timestamp", 0)) / 86400
            total_size_mb += info.get("size_mb", 0)
            
            # 删除超过 7 天的缓存
            if age_days > _MAX_CACHE_AGE_DAYS:
                to_remove.append(key)
        
        # 如果总大小超过限制，删除最旧的缓存
        while total_size_mb > _MAX_CACHE_SIZE_GB * 1024 and len(self._cache_index) > len(to_remove):
            oldest_key = next(iter(self._cache_index))
            if oldest_key not in to_remove:
                to_remove.append(oldest_key)
                total_size_mb -= self._cache_index[oldest_key].get("size_mb", 0)
        
        # 执行删除
        for key in to_remove:
            entry = self._cache_index.get(key, {})
            cache_path = Path(entry.get("path", ""))
            if cache_path.exists():
                cache_path.unlink()
            del self._cache_index[key]
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个过期缓存")
    
    def clear(self) -> None:
        """清空所有缓存。"""
        for info in self._cache_index.values():
            cache_path = Path(info["path"])
            if cache_path.exists():
                cache_path.unlink()
        self._cache_index.clear()
        self._save_index()
        logger.info("流水线缓存已清空")


# ══════════════════════════════════════════════════════════════════════════════
# 参数与结果 (优化版)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipelineParams:
    """流水线参数（不可变）。"""
    test_size: float = 0.2
    random_seed: int = 42
    window_size: int = 200
    step_size: int = 100
    pca_variance: float = 0.9019
    use_pca: bool = True
    max_files_per_subject: int | None = None
    feature_groups: tuple[str, ...] = ("spectral", "statistical", "temporal")

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 0.5:
            raise ValueError(f"test_size ∈ (0, 0.5): {self.test_size}")
        if self.window_size < 2:
            raise ValueError(f"window_size > 1: {self.window_size}")

    @classmethod
    def from_config(cls, **overrides):
        cfg = PipelineConfig.from_root()
        return cls(**{n: overrides.get(n, getattr(cfg, n)) for n in _PARAM_NAMES})

    def to_dict(self) -> dict:
        d = {n: getattr(self, n) for n in _PARAM_NAMES}
        if self.max_files_per_subject is not None:
            d["max_files"] = self.max_files_per_subject
        return d

    @property
    def hash(self) -> str:
        """生成参数哈希，用于缓存键。"""
        import hashlib
        params_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]


@dataclass
class PipelineResult:
    """流水线执行结果 — 增加性能分析。"""
    params: PipelineParams
    split_meta: dict
    window_meta: dict
    split_file: Path
    window_file: Path
    process_meta: dict | None = None
    processed_file: Path | None = None
    model_metrics: dict = field(default_factory=dict)
    model_type: ModelType = "svm"
    elapsed_s: float = 0.0
    # 新增：性能分析
    stage_times: dict = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "model_type": self.model_type,
            "params": self.params.to_dict(),
            "split_meta": self.split_meta,
            "window_meta": self.window_meta,
            "split_file": str(self.split_file),
            "window_file": str(self.window_file),
            "model_metrics": self.model_metrics,
            "elapsed_s": self.elapsed_s,
            "stage_times": self.stage_times,
            "memory_usage_mb": self.memory_usage_mb,
        }
        if self.process_meta:
            d["process_meta"] = self.process_meta
            d["processed_file"] = str(self.processed_file)
        return d


def _restore_from_cache(file_path: Path, data: dict) -> None:
    """从缓存恢复中间文件到磁盘。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"从缓存恢复: {file_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 数据预处理流水线 (RSSI) 
# ══════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """RSSI 数据预处理流水线: 划分 → 窗口 → 特征。
    
    优化:
      - 断点续传：跳过已完成的阶段
      - 缓存复用：同一参数不重复计算
      - 内存监控：自动降级
    """

    def __init__(self, params: PipelineParams | None = None, context: DataLoadContext | None = None):
        self.params = params or PipelineParams()
        self.pcfg = PipelineConfig.from_root()
        self.cache = PipelineCache(self.pcfg)
        self.memory_monitor = get_memory_monitor().spawn()
        self._load_ctx = context or DataLoadContext.global_instance()

    def _get_paths(self, task: TaskType) -> tuple[Path, Path, Path]:
        bd = self.pcfg.data_dir
        mf = self.params.max_files_per_subject
        h = self.params.hash
        suffix = f"_mf{mf}" if mf is not None else ""
        return (
            bd / f"rssi_split_{task}_{h}{suffix}.pkl",
            bd / f"rssi_windowed_{task}_{h}{suffix}.pkl",
            bd / f"rssi_processed_{task}_{h}{suffix}.pkl",
        )

    def run(self, task: TaskType, skip_features: bool = False,
            compress: bool = False, use_cache: bool = True) -> PipelineResult:
        """运行数据预处理流水线。
        
        Args:
            task: 任务类型
            skip_features: 是否跳过特征提取（CNN 模式）
            compress: 是否压缩中间文件
            use_cache: 是否使用缓存
            
        Returns:
            PipelineResult 包含所有阶段的元数据
        """
        t0 = time.time()
        stage_times = {}
        
        # 检查内存压力
        self.memory_monitor.log_status()
        if self.memory_monitor.should_degrade():
            logger.warning("检测到高内存压力，将使用保守策略")
            compress = True  # 压缩以减少磁盘和内存使用
        
        split_f, win_f, proc_f = self._get_paths(task)
        
        # ── 阶段 1: 数据划分 ──
        t1 = time.time()
        params_hash = self.params.hash
        split_r = None

        if use_cache and self.cache.exists("split", f"{task}_{params_hash}"):
            split_r = self.cache.load("split", f"{task}_{params_hash}")
            if split_r is not None:
                logger.info(f"复用缓存的划分结果")
                if not split_f.exists():
                    _restore_from_cache(split_f, split_r)

        if split_r is None and split_f.exists():
            logger.info(f"加载已有划分: {split_f}")
            with split_f.open("rb") as f:
                split_r = pickle.load(f)

        if split_r is None:
            splitter = DatasetSplitter(
                seed=self.params.random_seed,
                max_files_per_subject=self.params.max_files_per_subject,
                context=self._load_ctx,
            )
            split_r = splitter.split_authentication(
                self.params.test_size, compress=compress)

            if "auth_test" in split_r:
                split_r["test"] = split_r.pop("auth_test")

            with split_f.open("wb") as f:
                pickle.dump(split_r, f, protocol=pickle.HIGHEST_PROTOCOL)

            if use_cache:
                self.cache.save("split", f"{task}_{params_hash}", split_r)

        if split_r is None:
            raise RuntimeError(
                f"数据划分失败：缓存已过期或文件损坏。"
                f"请清除缓存后重试: rm -rf {self.pcfg.cache_dir}/*")

        stage_times["split"] = time.time() - t1
        logger.info(f"划分完成 → {split_f} ({stage_times['split']:.1f}s)")
        
        gc.collect()

        # ── 阶段 2: 窗口构建 ──
        t2 = time.time()
        win_r = None

        if use_cache and self.cache.exists("window", f"{task}_{params_hash}"):
            win_r = self.cache.load("window", f"{task}_{params_hash}")
            if win_r is not None:
                logger.info(f"复用缓存的窗口结果")
                if not win_f.exists():
                    _restore_from_cache(win_f, win_r)

        if win_r is None and win_f.exists():
            logger.info(f"加载已有窗口: {win_f}")
            with win_f.open("rb") as f:
                win_r = pickle.load(f)

        if win_r is None:
            win_r = WindowProcessor(
                WindowConfig(self.params.window_size, self.params.step_size),
            ).process(split_file=split_f, output_file=win_f, compress=compress)

            if use_cache:
                self.cache.save("window", f"{task}_{params_hash}", win_r, compress)

        if win_r is None:
            raise RuntimeError(
                f"窗口构建失败：缓存已过期或文件损坏。"
                f"请清除缓存后重试: rm -rf {self.pcfg.cache_dir}/*")

        stage_times["window"] = time.time() - t2
        logger.info(f"窗口构建完成 ({stage_times['window']:.1f}s)")

        # ── 阶段 3: 特征提取 ──
        proc_meta = None
        if not skip_features:
            t3 = time.time()
            proc_r = None

            if use_cache and self.cache.exists("features", f"{task}_{params_hash}"):
                proc_r = self.cache.load("features", f"{task}_{params_hash}")
                if proc_r is not None:
                    proc_meta = proc_r["meta"]
                    logger.info(f"复用缓存的特征结果")
                    if not proc_f.exists():
                        _restore_from_cache(proc_f, proc_r)

            if proc_r is None and proc_f.exists():
                logger.info(f"加载已有特征: {proc_f}")
                with proc_f.open("rb") as f:
                    proc_r = pickle.load(f)
                proc_meta = proc_r["meta"]

            if proc_r is None:
                proc_r = FeatureProcessor(
                    PreprocessConfig(
                        use_pca=self.params.use_pca,
                        pca_variance=self.params.pca_variance,
                        feature_groups=self.params.feature_groups,
                    ),
                ).process(input_file=win_f, output_file=proc_f)
                proc_meta = proc_r["meta"]

                if use_cache:
                    self.cache.save("features", f"{task}_{params_hash}", proc_r)

            if proc_r is None:
                raise RuntimeError(
                    f"特征提取失败：缓存已过期或文件损坏。"
                    f"请清除缓存后重试: rm -rf {self.pcfg.cache_dir}/*")

            stage_times["features"] = time.time() - t3
            logger.info(f"特征提取完成 ({stage_times['features']:.1f}s)")
        
        memory_usage = 0.0
        if hasattr(self, '_get_memory_usage'):
            memory_usage = self._get_memory_usage() # type: ignore[attr-defined]
        
        elapsed = time.time() - t0
        logger.info(f"数据流水线完成 ({elapsed:.1f}s)")
        
        return PipelineResult(
            self.params, 
            split_r.get("meta", {}) if not skip_features else {},
            win_r.get("meta", {}) if not skip_features else {},
            split_f, win_f, proc_meta,
            proc_f if not skip_features else None,
            elapsed_s=elapsed,
            stage_times=stage_times,
            memory_usage_mb=memory_usage,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CSI 数据协调器 — 将数据准备从 AuthPipeline 分离
# ══════════════════════════════════════════════════════════════════════════════

class DataCoordinator:
    """CSI 数据协调器 — 负责 CSI 数据的划分、窗口构建、特征提取。

    从 AuthPipeline 中分离数据准备逻辑，使流水线类专注于工作流编排。
    """

    def __init__(self, pipeline: "AuthPipeline"):
        self._p = pipeline

    # ── 属性代理 ──
    @property
    def _params_hash(self): return self._p._params_hash
    @property
    def pcfg(self): return self._p.pcfg
    @property
    def cache(self): return self._p.cache
    @property
    def use_cache(self): return self._p.use_cache
    @property
    def clean_intermediate(self): return self._p.clean_intermediate
    @property
    def memory_monitor(self): return self._p.memory_monitor
    @property
    def seed(self): return self._p.seed
    @property
    def test_size(self): return self._p.test_size
    @property
    def window_size(self): return self._p.window_size
    @property
    def step_size(self): return self._p.step_size
    @property
    def use_pca(self): return self._p.use_pca
    @property
    def pca_variance(self): return self._p.pca_variance
    @property
    def max_files(self): return self._p.max_files
    @property
    def cross_activity(self): return self._p.cross_activity
    @property
    def _progress_cb(self): return self._p._progress_cb

    # ── SVM 数据准备 ──

    def prepare_svm(self) -> tuple:
        """为 SVM 认证准备 CSI 数据: 划分 → 窗口 → 特征。"""
        logger.info("CSI 数据准备 (优化模式)...")

        if self.use_cache:
            cache_key = f"csi_prepared_{self._params_hash}"
            cached = self.cache.load("csi_data", cache_key)
            if cached is not None:
                logger.info("复用缓存的 CSI 数据")
                fe = FeatureExtractor(PreprocessConfig(
                    use_pca=self.use_pca, pca_variance=self.pca_variance,
                    feature_groups=self._p.feature_groups))
                fe.pca = cached.get("pca_model")
                fe.scaler = cached.get("scaler_model")
                self._p._fe = fe
                return (cached["x_train"].astype(np.float32), cached["y_train"],
                        cached["x_test"].astype(np.float32), cached["y_test"],
                        Path(cached["data_file"]), fe)

        sr = self._load_or_create_split()
        fe = FeatureExtractor(PreprocessConfig(
            use_pca=self.use_pca, pca_variance=self.pca_variance,
            feature_groups=self._p.feature_groups))
        wb = WindowBuilder(WindowConfig(self.window_size, self.step_size))

        tr_feats, y_tr = self._batch_extract(sr["train"], wb, fe, "train", _CSI_LOG_INTERVAL_SVM)
        if self.clean_intermediate:
            del sr["train"]; gc.collect()
        te_feats, y_te = self._batch_extract(sr["auth_test"], wb, fe, "test", _CSI_LOG_INTERVAL_SVM)
        del sr; gc.collect()

        logger.info(f"特征: 训练 {tr_feats.shape}, 测试 {te_feats.shape}")
        # 对已提取的特征做 PCA → Scaler (使用底层方法，避免 transform() 的二阶特征提取)
        tr_feats = fe.fit_pca(tr_feats)
        te_feats = fe.transform_pca(te_feats)
        tr_feats = fe.fit_scaler(tr_feats)
        te_feats = fe.transform_scaler(te_feats)

        data_file = self.pcfg.data_dir / f"rssi_processed_authentication_npy_{self._params_hash}.pkl"
        with data_file.open("wb") as f:
            pickle.dump({"x_train": tr_feats, "y_train": y_tr,
                         "x_test": te_feats, "y_test": y_te,
                         "pca_model": fe.pca, "scaler_model": fe.scaler},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.use_cache:
            self.cache.save("csi_data", f"csi_prepared_{self._params_hash}", {
                "x_train": tr_feats, "y_train": y_tr,
                "x_test": te_feats, "y_test": y_te,
                "pca_model": fe.pca, "scaler_model": fe.scaler,
                "data_file": str(data_file),
            })
        self._p._fe = fe
        return (tr_feats, y_tr, te_feats, y_te, data_file, fe)

    # ── CNN 数据准备 ──

    def prepare_cnn(self) -> Path:
        """为 CNN 认证准备 CSI 数据: 划分 → 流式窗口 → memmap。"""
        cache_key = f"cnn_csi_windows_{self._params_hash}"
        if self.use_cache and self.cache.exists("cnn_windows", cache_key):
            cached = self.cache.load("cnn_windows", cache_key)
            if cached is not None:
                y_tr = cached.get("y_train")
                if y_tr is not None and len(set(y_tr)) >= 2:
                    logger.info("复用缓存的 CNN 窗口数据 (有效, %d 用户)", len(set(y_tr)))
                    data_file = self.pcfg.data_dir / f"rssi_windowed_authentication_npy_{self._params_hash}.pkl"
                    with data_file.open("wb") as f:
                        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
                    return data_file
                logger.warning("缓存的 CNN 窗口数据无效 (<2 用户), 重新构建")

        splitter = DatasetSplitter(seed=self.seed, data_source="csi",
                                   max_files_per_subject=self.max_files,
                                   context=self._p._load_ctx)
        sr = splitter.split_authentication(test_size=self.test_size,
                                           cross_activity=self.cross_activity)
        wb = WindowBuilder(WindowConfig(self.window_size, self.step_size))
        tr_path, y_tr, n_tr, n_channels = self.build_windows_memmap(sr["train"], wb, "train")
        _clear_sample_cache(self._p._load_ctx); gc.collect()
        te_path, y_te, n_te, _ = self.build_windows_memmap(sr["auth_test"], wb, "test")
        del sr; gc.collect()
        logger.info(f"窗口: 训练 {n_tr}, 测试 {n_te}")

        window_data = {"x_train": str(tr_path), "y_train": y_tr,
                       "x_test": str(te_path), "y_test": y_te,
                       "is_memmap": True,
                       "x_train_shape": (n_tr, n_channels, self.window_size),
                       "x_test_shape": (n_te, n_channels, self.window_size)}
        data_file = self.pcfg.data_dir / f"rssi_windowed_authentication_npy_{self._params_hash}.pkl"
        with data_file.open("wb") as f:
            pickle.dump(window_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"CSI 窗口索引已保存: {data_file}")
        if self.use_cache:
            self.cache.save("cnn_windows", cache_key, window_data)
        return data_file

    # ── 内部工具 ──

    def _load_or_create_split(self) -> dict:
        cache_split = self.pcfg.data_dir / f"rssi_split_authentication_npy_{self._params_hash}.pkl"
        if cache_split.exists() and self.use_cache:
            logger.info(f"使用缓存划分: {cache_split}")
            with cache_split.open("rb") as f:
                return pickle.load(f)
        splitter = DatasetSplitter(seed=self.seed, data_source="csi",
                                   max_files_per_subject=self.max_files,
                                   context=self._p._load_ctx)
        n = len(splitter.samples)
        n_subj = len({s.subject for s in splitter.samples})
        logger.info(f"CSI: {n} 样本 ({n_subj} 用户)")
        sr = splitter.split_authentication(test_size=self.test_size,
                                           cross_activity=self.cross_activity)
        with cache_split.open("wb") as f:
            pickle.dump(sr, f, protocol=pickle.HIGHEST_PROTOCOL)
        del splitter; gc.collect()
        m = sr["meta"]
        logger.info(f"划分: 训练 {m['num_train_samples']}, 测试 {m['num_auth_test_samples']}")
        return sr

    def _process_one_sample(self, s, wb, fe):
        """处理单个样本: 加载 → 降噪 → 滑窗 → 特征提取 (线程安全)。"""
        from scripts.data_loader import load_npy_matrix
        raw = s.get("data")
        if raw is None: return None
        if isinstance(raw, Path):
            try: arr = load_npy_matrix(raw)
            except (ValueError, OSError): return None
        else:
            arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2: return None
        # CSI 原始信号降噪 (窗口构建前)
        denoise_method = self._p._csi_denoise or self._p.pcfg.csi_denoise
        arr = _apply_csi_denoise(arr, denoise_method)
        windows = wb.build(arr)
        if windows.shape[0] == 0: return None
        feats = fe.extract_features(windows)
        label = s.get("subject", s.get("claimed_identity", "?"))
        return (feats, [label] * windows.shape[0])

    def _batch_extract(self, samples, wb, fe, name, log_interval):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        total = len(samples)
        cb = self._progress_cb
        merged_feats, merged_labels = [], []
        n_done = 0

        max_workers = min(8, (getattr(self, 'pcfg', None) is not None and 8) or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._process_one_sample, s, wb, fe): i
                      for i, s in enumerate(samples)}
            for fut in as_completed(futures):
                n_done += 1
                if n_done % log_interval == 0 or n_done == 1 or n_done == total:
                    logger.debug(f"  [{name}] {n_done}/{total}")
                    if cb: cb(n_done, total, f"特征提取 ({name}): {n_done}/{total}")
                try:
                    rv = fut.result()
                except Exception:
                    continue
                if rv is None: continue
                feats, labels = rv
                merged_feats.append(feats)
                merged_labels.extend(labels)

        total_windows = sum(f.shape[0] for f in merged_feats)
        logger.info(f"  [{name}] {total}/{total} → {total_windows} 窗口")
        if not merged_feats:
            return (np.empty((0, 1), dtype=np.float32), [])
        return np.concatenate(merged_feats, axis=0), merged_labels

    def build_windows_memmap(self, samples, wb, name):
        from scripts.data_loader import load_npy_matrix
        total = len(samples); all_labels = []
        cb = self._progress_cb
        n_channels, total_windows = 0, 0

        # 第一遍: 仅读取文件头获取 shape，不加载数据
        # 每 ~100 文件报告进度, 避免用户感知卡顿
        _HEADER_SCAN_REPORT = 100
        for i, s in enumerate(samples):
            if cb and i % _HEADER_SCAN_REPORT == 0:
                cb(i, total, f"扫描窗口 ({name}): {i}/{total}")
            raw = s.get("data")
            if raw is None: continue
            if isinstance(raw, Path):
                try:
                    fp = str(raw)
                    mm = np.load(fp, mmap_mode='r')
                    arr_shape = mm.shape
                    if hasattr(mm, '_mmap'):
                        mm._mmap.close()
                except (ValueError, OSError): continue
            else:
                arr_shape = np.asarray(raw).shape
            if len(arr_shape) != 2: continue
            # NPY 原始格式 (n_subcarriers, n_timesteps),
            # load_npy_matrix 会 transpose 为 (n_timesteps, n_subcarriers)
            n_steps = arr_shape[1]  # 时间步 (axis=1 = timesteps)
            n_feat = arr_shape[0]   # 子载波数 (axis=0 = subcarriers)
            if n_channels == 0:
                n_channels = n_feat
                logger.info(f"  [{name}] 检测到 {n_channels} 通道, {n_steps} 时间步")
            n_w = wb.estimate_num_windows(n_steps, self.window_size, self.step_size)
            total_windows += n_w
            subj = s.get("subject", s.get("claimed_identity", "?"))
            all_labels.extend([subj] * n_w)
        if cb:
            cb(total, total, f"扫描窗口 ({name}): {total}/{total}")

        estimated_gb = (total_windows * n_channels * self.window_size * 4) / (1024**3)
        logger.info(f"  [{name}] 总计: {total_windows} 窗口, 预估 {estimated_gb:.1f} GiB")
        if total_windows == 0:
            empty_path = self.pcfg.cache_dir / f"csi_windows_{name}_empty.npy"
            np.save(empty_path, np.empty((0, n_channels, self.window_size), dtype=np.float32))
            return empty_path, np.array([], dtype=object), 0, n_channels
        if estimated_gb > 10 and self.memory_monitor.should_degrade():
            raise RuntimeError(f"预估内存需求 {estimated_gb:.1f} GiB 过大")

        # 第二遍: 加载数据、构建窗口、写入 memmap
        mmap_path = self.pcfg.cache_dir / f"csi_windows_{name}.dat"
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        if mmap_path.exists(): mmap_path.unlink()
        shape = (total_windows, n_channels, self.window_size)
        logger.info(f"  [{name}] 创建内存映射文件: {mmap_path} ({shape})")
        mmap = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)
        write_pos = 0
        for i, s in enumerate(samples):
            if (i + 1) % _CSI_LOG_INTERVAL_CNN == 0 or i == 0:
                logger.debug(f"  [{name}] 写入: {i + 1}/{total} → {write_pos}/{total_windows}")
            raw = s.get("data")
            if raw is None: continue
            if isinstance(raw, Path):
                try: arr = load_npy_matrix(raw)
                except (ValueError, OSError): continue
            else: arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim != 2: continue
            # CSI 原始信号降噪 (窗口构建前, 保留完整时间上下文)
            denoise_method = self._p._csi_denoise or self._p.pcfg.csi_denoise
            arr = _apply_csi_denoise(arr, denoise_method)
            windows = wb.build(arr)
            if windows.shape[0] == 0: continue
            # wb.build 返回 (n, n_subcarriers, window_size)
            # memmap shape 同为 (total, n_subcarriers, window_size) → 直接写入
            mmap[write_pos:write_pos + windows.shape[0]] = windows
            write_pos += windows.shape[0]

        mmap.flush()
        # 显式释放 memmap, 避免 Windows 文件锁阻止后续 mode='r+' 打开
        if hasattr(mmap, '_mmap'):
            mmap._mmap.close()
        del mmap
        logger.info(f"  [{name}] 写入完成: {write_pos} 窗口 → {mmap_path}")
        return mmap_path, np.array(all_labels, dtype=object), total_windows, n_channels

    def cleanup(self) -> None:
        """清理中间文件并同步清理缓存索引。"""
        patterns = [self.pcfg.cache_dir / "csi_windows_*.dat",
                    self.pcfg.cache_dir / "csi_windows_*_empty.npy"]
        deleted = False
        for pattern in patterns:
            for file_path in self.pcfg.cache_dir.glob(pattern.name):
                try:
                    file_path.unlink()
                    logger.debug(f"已清理: {file_path}")
                    deleted = True
                except Exception as e:
                    logger.warning(f"清理失败: {file_path} - {e}")
        if deleted and self.use_cache:
            to_remove = []
            for key, info in self.cache._cache_index.items():
                path = info.get("path", "")
                if isinstance(path, str) and "cnn_csi_windows" in path:
                    to_remove.append(key)
            for key in to_remove:
                self.cache._cache_index.pop(key, None)
                logger.debug(f"已清理过期缓存: {key}")
            if to_remove: self.cache._save_index()


# ══════════════════════════════════════════════════════════════════════════════
# 统一认证流水线 — RSSI + CSI
# ══════════════════════════════════════════════════════════════════════════════

class AuthPipeline:
    """统一身份认证流水线 — RSSI (MAT) 和 CSI (NPY) 通用。
    
    优化:
      - 流式窗口构建，避免 OOM
      - 内存映射文件自动管理
      - 分批特征提取，控制内存峰值
      - 训练后自动清理中间文件
      - v2.3: 模型缓存受 use_model_cache 严格约束，参数变化触发重新训练
    """

    def __init__(
        self, *,
        data_source: DataSource = "rssi",
        seed: int = 42,
        test_size: float = 0.2,
        window_size: int = 200,
        step_size: int = 100,
        pca_variance: float = 0.9019,
        use_pca: bool = True,
        threshold_method: str = "youden",
        cv_folds: int = 5,
        max_files_per_subject: int | None = 220,
        cross_activity: bool = False,
        clean_intermediate: bool = True,
        use_cache: bool = True,
        use_model_cache: bool | None = None,
        cache_path: str | Path | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        save_model: bool = True,
        save_metrics: bool = True,
        use_online_svm: bool = False,
        online_kernel: str = "linear",
        feature_groups: tuple[str, ...] = ("spectral", "statistical", "temporal"),
        csi_denoise: str | None = None,
    ):
        self.pcfg = PipelineConfig.from_root()
        # 运行时 CSI 降噪覆盖 (UI 选择优先于配置文件默认值)
        self._csi_denoise = csi_denoise

        if data_source == "csi" and max_files_per_subject is not None:
            na = len(self.pcfg.csi_selected_actions) if self.pcfg.csi_selected_actions else 55
            if max_files_per_subject % na != 0:
                raise ValueError(
                    f"CSI 每用户文件数必须为 {na} 的倍数: {max_files_per_subject}")

        self.source = data_source
        self.seed = seed
        self.test_size = test_size
        self.window_size = window_size
        self.step_size = step_size
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.threshold_method = threshold_method
        self.cv_folds = cv_folds
        self.max_files = max_files_per_subject
        self.cross_activity = cross_activity
        self.clean_intermediate = clean_intermediate
        self.use_cache = use_cache
        # 关键：use_model_cache 独立于 use_cache，默认跟随
        self.use_model_cache = use_cache if use_model_cache is None else use_model_cache
        self._progress_cb = progress_callback
        self.cache = PipelineCache(self.pcfg)
        self._cache_path_override = None
        if cache_path is not None:
            cp = Path(cache_path)
            cp.mkdir(parents=True, exist_ok=True)
            self.cache.cache_dir = cp
            self._cache_path_override = cp
        self.memory_monitor = get_memory_monitor().spawn()
        self._fe: FeatureExtractor | None = None
        self.save_model = save_model
        self.save_metrics = save_metrics
        self.use_online_svm = use_online_svm
        self.online_kernel = online_kernel
        self.feature_groups = feature_groups
        self._coordinator = DataCoordinator(self)
        self._params_hash = self._compute_params_hash()
        self._load_ctx = DataLoadContext()

    def _compute_params_hash(self) -> str:
        """计算当前参数的哈希值。
        
        注意：use_cache 和 use_model_cache 不参与哈希计算，
        因为它们是缓存策略而非模型参数。
        max_files 参与哈希以确保不同样本数不会命中同一缓存。
        """
        import hashlib
        # 读取 csi_selected_actions 参与哈希, 避免动作筛选变化后命中旧缓存
        csi_actions = None
        if self.source == "csi":
            pcfg = PipelineConfig.from_root()
            csi_actions = pcfg.csi_selected_actions
        params = {
            "source": self.source,
            "seed": self.seed,
            "test_size": self.test_size,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "use_pca": self.use_pca,
            "pca_variance": self.pca_variance,
            "threshold_method": self.threshold_method,
            "max_files": self.max_files,
            "cross_activity": self.cross_activity,
            "use_online_svm": self.use_online_svm,
            "online_kernel": self.online_kernel,
            "feature_groups": self.feature_groups,
            "csi_selected_actions": csi_actions,
            "csi_denoise": self._csi_denoise,
            "_cache_fmt": 2,  # 缓存格式版本, 递增以强制失效旧缓存
        }
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def build_log_config(self, **extra) -> dict:
        """构建完整的结构化日志配置 dict，统一 SVM/CNN/实验的日志输出。"""
        c = {
            "data_source": self.source,
            "seed": self.seed,
            "test_size": self.test_size,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "threshold_method": self.threshold_method,
            "cv_folds": self.cv_folds,
            "max_files_per_subject": self.max_files,
            "cross_activity": self.cross_activity,
            "use_pca": self.use_pca,
            "feature_groups": list(self.feature_groups),
            "csi_denoise": self._csi_denoise,
            "clean_intermediate": self.clean_intermediate,
        }
        if self.use_pca:
            c["pca_variance"] = self.pca_variance
        if self.use_online_svm or self.source == "csi":
            c["use_online_svm"] = self.use_online_svm
            c["online_kernel"] = self.online_kernel
        c.update(extra)
        return c

    # ══════════════════════════════════════════════════════════════════════
    # 数据准备（优化版）
    # ══════════════════════════════════════════════════════════════════════

    def _prepare(self) -> tuple:
        """统一数据准备入口 — 支持缓存复用。"""
        if self.source == "rssi":
            return self._prepare_rssi()
        return self._prepare_csi()

    def _prepare_rssi(self) -> tuple:
        """RSSI 数据准备 — 使用 DataPipeline。"""
        dp = DataPipeline(PipelineParams(
            test_size=self.test_size, random_seed=self.seed,
            window_size=self.window_size, step_size=self.step_size,
            pca_variance=self.pca_variance, use_pca=self.use_pca,
            max_files_per_subject=self.max_files,
            feature_groups=self.feature_groups),
            context=self._load_ctx)
        rr = dp.run("authentication", use_cache=self.use_cache)

        if rr.processed_file is None:
            raise RuntimeError("数据预处理失败：processed_file 为 None")

        with rr.processed_file.open("rb") as f:
            d = pickle.load(f)

        return (
            np.asarray(d["x_train"], np.float32),
            d["y_train"],
            np.asarray(d["x_test"], np.float32),
            d["y_test"],
            rr.processed_file,
        )

    def _prepare_csi(self) -> tuple:
        """CSI 数据准备 — 委托给 DataCoordinator。"""
        return self._coordinator.prepare_svm()

    # ══════════════════════════════════════════════════════════════════════
    # SVM 认证
    # ══════════════════════════════════════════════════════════════════════

    def run_svm(self) -> dict:
        """SVM 认证流水线 — 支持缓存和内存优化。"""
        banner = f"{'=' * 60}\n{'RSSI' if self.source == 'rssi' else 'CSI'} SVM 认证流水线\n{'=' * 60}"
        logger.info(banner)
        
        # 检查模型缓存 —— 仅在 use_model_cache 为 True 时
        model_cache_key = f"svm_auth_{self._params_hash}"
        if self.use_model_cache and self.cache.exists("svm_model", model_cache_key):
            logger.info("复用缓存的 SVM 模型 (hash=%s)", self._params_hash)
            cached = self.cache.load("svm_model", model_cache_key)
            if cached is None:
                raise RuntimeError("SVM 模型缓存加载失败")
            _log_auth_summary(cached)
            return cached

        logger.info("执行完整 SVM 训练 (hash=%s, use_model_cache=%s)",
                     self._params_hash, self.use_model_cache)

        result_tuple = self._prepare()
        if len(result_tuple) == 5:
            x_tr, y_tr, x_te, y_te, data_file = result_tuple
            fe = None
        else:
            x_tr, y_tr, x_te, y_te, data_file, fe = result_tuple

        logger.info("训练 SVM 认证模型...")
        sc = SVMConfig(
            threshold_method=self.threshold_method,
            random_seed=self.seed,
            cv_folds=self.cv_folds,
        )
        suffix = "_npy" if self.source == "csi" else ""
        svm_path = self.pcfg.model_dir / f"svm_authentication{suffix}.pkl"
        t0 = time.time()
        try:
            result = SVMAuthenticationTrainer(
                sc, self.pcfg,
                use_online=self.use_online_svm,
                online_kernel=self.online_kernel,
            ).train(
                data_file=data_file, model_path=svm_path,
                data_source=self.source)
            log_training(
                model_type="SVM", task_type="authentication",
                data_source=self.source, status="success",
                duration=time.time() - t0,
                config=self.build_log_config(),
                metrics=result.get("system_metrics"),
                pipeline_config=self.pcfg,
            )
        except Exception:
            log_training(
                model_type="SVM", task_type="authentication",
                data_source=self.source, status="failed",
                duration=time.time() - t0,
                config=self.build_log_config(),
                error=str(_current_exception()),
                pipeline_config=self.pcfg,
            )
            raise

        if self.save_model:
            self._save_model(result, "svm", fe)
        if self.save_metrics:
            self._save_metrics(result, "svm")
        _log_auth_summary(result)
        
        # 缓存模型结果
        if self.use_model_cache:
            self.cache.save("svm_model", model_cache_key, result)

        # 清理中间文件
        if self.clean_intermediate:
            self._cleanup_intermediate_files()
        
        return result

    # ══════════════════════════════════════════════════════════════════════
    # CNN 认证
    # ══════════════════════════════════════════════════════════════════════

    def run_cnn(
        self,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        cancel_fn: Callable[[], None] | None = None,
        use_checkpoint: bool = True,
        gradient_accumulation_steps: int = 4,
        conv_channels: tuple[int, ...] | None = None,
        hidden_units: int | None = None,
    ) -> dict:
        """CNN 认证流水线 — 支持架构和训练超参数覆盖。"""
        banner = f"{'=' * 60}\n{'RSSI' if self.source == 'rssi' else 'CSI'} CNN 认证流水线\n{'=' * 60}"
        logger.info(banner)

        # 检查模型缓存 —— 仅在 use_model_cache 为 True 时
        cc_str = "_".join(str(c) for c in conv_channels) if conv_channels else "def"
        model_cache_key = (
            f"cnn_auth_{self._params_hash}_{epochs}_{batch_size}"
            f"_{use_checkpoint}_{gradient_accumulation_steps}"
            f"_cc{cc_str}_hu{hidden_units or 'def'}")
        if self.use_model_cache and self.cache.exists("cnn_model", model_cache_key):
            logger.info("复用缓存的 CNN 模型 (hash=%s)", self._params_hash)
            cached = self.cache.load("cnn_model", model_cache_key)
            if cached is None:
                raise RuntimeError("CNN 模型缓存加载失败")
            _log_auth_summary(cached)
            return cached

        logger.info("执行完整 CNN 训练 (hash=%s, use_model_cache=%s)",
                     self._params_hash, self.use_model_cache)

        if self.source == "rssi":
            data_file = self._prepare_cnn_rssi()
        else:
            data_file = self._prepare_cnn_csi()

        # 内存监控
        self.memory_monitor.log_status()

        tc = CNNTrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold_method=self.threshold_method,
            random_seed=self.seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        suffix = "_npy" if self.source == "csi" else ""
        cnn_path = self.pcfg.model_dir / f"cnn_authentication{suffix}.pt"
        t0 = time.time()
        try:
            from scripts.models.cnn.models import CNNConfig as _CNNConfig
            mcfg_kw = {"use_checkpoint": use_checkpoint}
            if conv_channels is not None:
                mcfg_kw["conv_channels"] = conv_channels
            if hidden_units is not None:
                mcfg_kw["hidden_units"] = hidden_units
            model_cfg = _CNNConfig(**mcfg_kw)
            result = CNNTrainer(
                model_cfg=model_cfg,
                train_config=tc,
                config=self.pcfg,
                cancel_fn=cancel_fn,
            ).train_authentication(data_file, model_path=cnn_path)
            log_training(
                model_type="CNN", task_type="authentication",
                data_source=self.source, status="success",
                duration=time.time() - t0,
                config=self.build_log_config(
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    use_checkpoint=use_checkpoint,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                ),
                metrics=result.get("system_metrics"),
                pipeline_config=self.pcfg,
            )
        except Exception:
            log_training(
                model_type="CNN", task_type="authentication",
                data_source=self.source, status="failed",
                duration=time.time() - t0,
                config=self.build_log_config(
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    use_checkpoint=use_checkpoint,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                ),
                error=_current_exception(),
                pipeline_config=self.pcfg,
            )
            raise

        # 复制检查点
        ckpt = result.get("checkpoint_path", "")
        if ckpt:
            ckpt_path = Path(ckpt)
            suffix = "_npy" if self.source == "csi" else ""
            dest = self.pcfg.model_dir / f"cnn_authentication{suffix}.pt"
            if ckpt_path.resolve() != dest.resolve():
                self.pcfg.model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt_path, dest)
            logger.info(f"CNN 检查点: {dest}")

        if self.save_metrics:
            self._save_metrics(result, "cnn")
        _log_auth_summary(result)
        
        # 缓存模型结果
        if self.use_model_cache:
            self.cache.save("cnn_model", model_cache_key, result)

        # 清理 GPU 和中间文件
        clear_gpu_memory()
        if self.clean_intermediate:
            self._cleanup_intermediate_files()
        
        return result

    def _prepare_cnn_rssi(self) -> Path:
        """RSSI CNN 数据准备。"""
        dp = DataPipeline(PipelineParams(
            test_size=self.test_size, random_seed=self.seed,
            window_size=self.window_size, step_size=self.step_size,
            max_files_per_subject=self.max_files,
            feature_groups=self.feature_groups),
            context=self._load_ctx)
        rr = dp.run("authentication", skip_features=True, use_cache=self.use_cache)
        return rr.window_file

    def _prepare_cnn_csi(self) -> Path:
        """CSI CNN 数据准备 — 委托给 DataCoordinator。"""
        return self._coordinator.prepare_cnn()

    def _cleanup_intermediate_files(self) -> None:
        """清理中间文件 — 委托给 DataCoordinator。"""
        self._coordinator.cleanup()

    def _save_model(self, result: dict, model_type: str, fe=None) -> None:
        """保存模型 — 使用高效序列化。"""
        self.pcfg.model_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_npy" if self.source == "csi" else ""
        model = result.get("model")
        
        if model and model_type == "svm":
            model.pca_model = fe.pca if fe is not None else None
            model.scaler_model = fe.scaler if fe is not None else None
            path = self.pcfg.model_dir / f"svm_authentication{suffix}.pkl"
            with path.open("wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"SVM 模型已保存: {path}")

    def _save_metrics(self, result: dict, model_type: str) -> None:
        """保存评估指标。"""
        self.pcfg.result_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_npy" if self.source == "csi" else ""
        sm = result.get("system_metrics", {})

        path = self.pcfg.result_dir / f"authentication{suffix}_{model_type}_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "model": f"{'SVM' if model_type == 'svm' else 'CNN'} Auth",
                "data_source": self.source.upper(),
                "threshold_method": self.threshold_method,
                "params": {
                    "seed": self.seed, "test_size": self.test_size,
                    "window_size": self.window_size, "step_size": self.step_size,
                    "use_pca": self.use_pca, "cross_activity": self.cross_activity,
                },
                "system_metrics": {k: v for k, v in sm.items() if k != "user_metrics"},
                "user_metrics": sm.get("user_metrics", {}),
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"指标已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 工具函数 (优化版)
# ══════════════════════════════════════════════════════════════════════════════

def _build_overrides(**kw) -> dict:
    """构建参数覆盖字典 — 过滤 None 值。"""
    return {k: v for k, v in kw.items() if v is not None}


def _log_auth_summary(result: dict) -> None:
    """输出认证结果摘要 — 优化格式。"""
    sm = result.get("system_metrics", {})
    subjects = result.get("subjects", [])
    
    logger.info(f"{'=' * 60}")
    logger.info("认证结果汇总")
    logger.info(f"{'=' * 60}")
    logger.info(f"用户数: {len(subjects)}")
    
    mh = sm.get("mean_hter", "N/A")
    logger.info(f"HTER: {mh:.4f}" if isinstance(mh, float) else f"HTER: {mh}")
    logger.info(f"FAR:   {sm.get('mean_far', 'N/A')}")
    logger.info(f"FRR:   {sm.get('mean_frr', 'N/A')}")
    logger.info(f"准确率: {sm.get('global_accuracy', 'N/A')}")
    
    if um := sm.get("user_metrics", {}):
        logger.info("各用户性能 (前5个):")
        for i, (u, m) in enumerate(um.items()):
            if i >= 5:
                logger.info(f"  ... 还有 {len(um) - 5} 个用户")
                break
            logger.info(
                f"  {u}: HTER={m['hter']:.4f}, "
                f"FAR={m['far']:.4f}, FRR={m['frr']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数 (优化版)
# ══════════════════════════════════════════════════════════════════════════════

def run_authentication_pipeline(
    test_size=None, seed=None, window_size=None, step_size=None,
    pca_variance=None, use_pca=None, model_type: ModelType = "svm",
    use_cache: bool = True,
) -> dict:
    """认证流水线 — 支持缓存。"""
    p = AuthPipeline(
        data_source="rssi",
        test_size=test_size or 0.2, seed=seed or 42,
        window_size=window_size or 200, step_size=step_size or 100,
        pca_variance=pca_variance or 0.9019,
        use_pca=use_pca if use_pca is not None else True,
        use_cache=use_cache,
    )
    
    result = p.run_svm() if model_type == "svm" else p.run_cnn()
    return {"model_metrics": result.get("system_metrics", {})}


def run_data_pipeline(
    test_size=None, seed=None, window_size=None, step_size=None,
    pca_variance=None, use_pca=None,
    compress: bool = False, use_cache: bool = True,
) -> dict:
    """数据预处理流水线 (认证任务)。"""
    params = PipelineParams.from_config(**_build_overrides(
        test_size=test_size, random_seed=seed, window_size=window_size,
        step_size=step_size, pca_variance=pca_variance, use_pca=use_pca))

    d = DataPipeline(params).run(
        "authentication", compress=compress, use_cache=use_cache).to_dict()
    d.pop("model_metrics", None)
    d.pop("model_type", None)
    return d


def run_npy_authentication_svm(
    seed=42, test_size=0.2, window_size=200, step_size=100,
    pca_variance=0.9019, use_pca=True, threshold_method="youden",
    max_files_per_subject=220, cross_activity=False,
    use_cache: bool = True, clean_intermediate: bool = True,
) -> dict:
    """NPY SVM 认证流水线 — 优化版。"""
    mf = max_files_per_subject
    if get_memory_monitor().should_degrade():
        mf = min(mf, 110)
        logger.warning("内存压力高，降低 max_files_per_subject: %d → %d",
                       max_files_per_subject, mf)

    return AuthPipeline(
        data_source="csi", seed=seed, test_size=test_size,
        window_size=window_size, step_size=step_size,
        pca_variance=pca_variance, use_pca=use_pca,
        threshold_method=threshold_method,
        max_files_per_subject=mf,
        cross_activity=cross_activity,
        use_cache=use_cache,
        clean_intermediate=clean_intermediate,
    ).run_svm()


def run_npy_authentication_cnn(
    seed=42, test_size=0.2, window_size=200, step_size=100,
    threshold_method="youden", epochs=20, batch_size=64,
    learning_rate=0.001, max_files_per_subject=220,
    cross_activity=False,
    use_cache: bool = True, clean_intermediate: bool = True,
) -> dict:
    """NPY CNN 认证流水线 — 优化版。"""
    bs = batch_size
    optimal_batch = get_memory_monitor().get_optimal_batch_size(bs)
    if optimal_batch != bs:
        logger.info(f"批次大小调整: {bs} → {optimal_batch}")
        bs = optimal_batch

    return AuthPipeline(
        data_source="csi", seed=seed, test_size=test_size,
        window_size=window_size, step_size=step_size,
        threshold_method=threshold_method,
        max_files_per_subject=max_files_per_subject,
        cross_activity=cross_activity,
        use_cache=use_cache,
        clean_intermediate=clean_intermediate,
    ).run_cnn(epochs=epochs, batch_size=bs, learning_rate=learning_rate)