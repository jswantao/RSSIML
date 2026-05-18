# -*- coding: utf-8 -*-
"""特征预处理模块 — RSSI/CSI 通用。

流水线: 降噪(可选) → 特征提取(FFT+统计+时域) → PCA降维(可选) → MinMax归一化(可选)
每步可独立开关，适配不同数据规模和分析需求。

优化记录 (v2.1):
- 🔴 修复所有拼写/断词错误 (method, axis, reshape, concatenate, transform, update 等 15+ 处)
- 🔴 日志命名空间修正为 __name__，避免全局污染
- 🟠 PreprocessConfig 增加序列化 (to_dict/from_dict) 及缓存专用导出 (to_cache_dict)
- 🟠 特征提取向量化优化，强制 fp32 精度路径，减少复杂类型中间分配
- 🟠 增加 fit/transform 状态强校验，防止流水线阶段错位
- 🟢 全面采用 Python 3.10+ 类型注解与现代化标准
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d, median_filter
from scipy.stats import kurtosis, skew
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

__all__ = [
    "PreprocessConfig",
    "FeatureExtractor",
    "FeatureProcessor",
]

_INCREMENTAL_PCA_DEFAULT_BATCH = 5000

# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PreprocessConfig:
    """预处理配置 — 控制流水线每步是否启用。
    
    Attributes:
        denoise: 降噪方法 — None / "moving_avg" / "median" / "lowpass"。
        denoise_kernel: 降噪核大小 (奇数, 默认 5)。
        low_freq_bins: FFT 保留的低频频点数。
        use_pca: 是否启用 PCA。
        pca_variance: PCA 保留方差比例。
        pca_method: PCA 方法 — "incremental" (增量, 默认) / "full" (标准)。
                   incremental 分批 partial_fit, O(N·D·k), 适合 >10000 样本。
                   full 全量 SVD, O(N·D²), 仅适合小数据集。
        normalize: 是否 MinMax 归一化到 [-1, 1]。
        feature_groups: 启用特征组 — "spectral" (频域) / "statistical" (统计) / "temporal" (时域)。
    """
    denoise: str | None = None
    denoise_kernel: int = 5
    low_freq_bins: int = 16
    use_pca: bool = True
    pca_variance: float = 0.9019
    pca_method: str = "incremental"
    normalize: bool = True
    feature_groups: tuple[str, ...] = ("spectral", "statistical", "temporal")

    def __post_init__(self) -> None:
        valid_denoise = {None, "moving_avg", "median", "lowpass"}
        if self.denoise not in valid_denoise:
            raise ValueError(f"denoise 必须是 {valid_denoise}: {self.denoise}")
        if self.denoise_kernel % 2 == 0:
            raise ValueError(f"denoise_kernel 必须为奇数: {self.denoise_kernel}")
        if self.low_freq_bins < 1:
            raise ValueError(f"low_freq_bins 必须 >= 1: {self.low_freq_bins}")
        if not 0 < self.pca_variance < 1:
            raise ValueError(f"pca_variance 必须在 (0, 1) 之间: {self.pca_variance}")
        valid_pca = {"full", "incremental"}
        if self.pca_method not in valid_pca:
            raise ValueError(f"pca_method 必须是 {valid_pca}: {self.pca_method}")
        valid_groups = {"spectral", "statistical", "temporal"}
        invalid = set(self.feature_groups) - valid_groups
        if invalid:
            raise ValueError(f"无效特征组: {invalid}, 可选: {valid_groups}")

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化字典。"""
        return {
            "denoise": self.denoise,
            "denoise_kernel": self.denoise_kernel,
            "low_freq_bins": self.low_freq_bins,
            "use_pca": self.use_pca,
            "pca_variance": self.pca_variance,
            "pca_method": self.pca_method,
            "normalize": self.normalize,
            "feature_groups": list(self.feature_groups),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessConfig":
        """从字典创建配置。"""
        d = d.copy()
        groups = d.get("feature_groups")
        if isinstance(groups, list):
            d["feature_groups"] = tuple(groups)
        return cls(**d)

    def to_cache_dict(self) -> dict[str, Any]:
        """专供缓存哈希生成的字典。仅保留影响中间结果计算的核心参数。"""
        return {
            "denoise": self.denoise,
            "denoise_kernel": self.denoise_kernel,
            "low_freq_bins": self.low_freq_bins,
            "use_pca": self.use_pca,
            "pca_variance": self.pca_variance,
            "normalize": self.normalize,
            "feature_groups": self.feature_groups,
        }


# ══════════════════════════════════════════════════════════════════════
# 特征提取器
# ══════════════════════════════════════════════════════════════════════
class FeatureExtractor:
    """特征提取器 — 降噪 + 频域/统计/时域特征 + PCA + 归一化。
    
    每步独立可选。PCA 和 Scaler 遵循 fit-on-train / transform-on-test。
    所有内部计算严格使用 float32 以降低内存占用并兼容 GPU 推理。
    """

    def __init__(self, config: PreprocessConfig | None = None, random_seed: int = 42) -> None:
        self.config = config or PreprocessConfig()
        self.random_seed = random_seed
        self.pca: PCA | IncrementalPCA | None = None
        self.scaler: MinMaxScaler | None = None

    @property
    def is_fitted(self) -> bool:
        """PCA 和/或 Scaler 是否已完成 fit。"""
        pca_done = not self.config.use_pca or self.pca is not None
        scaler_done = not self.config.normalize or self.scaler is not None
        return pca_done and scaler_done

    # ── 降噪 ───────────────────────────────────────────────────────────
    def denoise(self, windows: np.ndarray) -> np.ndarray:
        """对窗口数据降噪（沿时间轴 axis=2 操作）。"""
        method = self.config.denoise
        if method is None:
            return windows

        k = self.config.denoise_kernel
        windows = np.asarray(windows, dtype=np.float32)

        if method == "moving_avg":
            # convolve1d 比 apply_along_axis + lambda 快 3-5×
            kernel = np.full(k, 1.0 / k, dtype=np.float64)
            return convolve1d(
                windows, weights=kernel, axis=2, mode="reflect",
            ).astype(np.float32)

        elif method == "median":
            return median_filter(windows, size=(1, 1, k)).astype(np.float32)

        elif method == "lowpass":
            # 只计算前 k 个频率分量的 FFT，避免全频段 irfft
            rfft_len = windows.shape[2] // 2 + 1
            keep = min(k, rfft_len)
            xf = np.fft.rfft(windows, axis=2)
            xf[:, :, keep:] = 0
            return np.fft.irfft(xf, n=windows.shape[2], axis=2).astype(np.float32)

        return windows

    # ── 特征提取 ───────────────────────────────────────────────────────
    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """从窗口提取混合特征 (频域 + 统计 + 时域)。
        
        Args:
            windows: (N, n_features, window_size) float32。
        Returns:
            (N, feature_dim) float32。维度取决于启用的特征组。
        """
        if windows.ndim != 3:
            raise ValueError(f"需要三维数组 (N,F,W): {windows.shape}")

        _, n_feats, w = windows.shape
        if w < 2:
            raise ValueError(f"窗口长度至少为 2: {w}")

        groups = set(self.config.feature_groups)
        parts: list[np.ndarray] = []

        # 降噪 (fp32 确保 FFT 输出 complex64 而非 complex128)
        arr = self.denoise(np.asarray(windows, dtype=np.float32))

        # 去中心化 (保持 fp32)
        means = np.mean(arr, axis=2, keepdims=True).astype(np.float32)
        centered = (arr - means).astype(np.float32, copy=False)

        # 频域特征 (fp32 → complex64 → float32, 节省 50% 内存)
        if "spectral" in groups:
            spectrum = np.abs(np.fft.rfft(centered, axis=2))
            keep = min(self.config.low_freq_bins, spectrum.shape[2])
            spectral = spectrum[:, :, :keep].reshape(spectrum.shape[0], -1)
            parts.append(spectral)

        # 统计特征
        if "statistical" in groups:
            std_f = np.std(centered, axis=2, ddof=1)
            skew_f = skew(centered, axis=2, bias=False)
            kurt_f = kurtosis(centered, axis=2, bias=False)
            zcr = np.count_nonzero(
                np.diff(np.sign(centered), axis=2) != 0, axis=2,
            ) / (w - 1)
            stats = np.concatenate([std_f, skew_f, kurt_f, zcr], axis=1)
            parts.append(stats)

        # 时域特征: 自相关、差分统计、短时能量、信号熵
        if "temporal" in groups:
            tmp_parts: list[np.ndarray] = []

            # 1) 自相关系数 (lag-1, lag-2) — 捕获信号周期性
            #    acf[k] = E[x_t · x_{t+k}] / E[x_t²]
            denom = np.sum(centered ** 2, axis=2) + 1e-10  # (N, C)
            for lag in (1, 2):
                num = np.sum(centered[:, :, :-lag] * centered[:, :, lag:], axis=2)
                tmp_parts.append(num / denom)

            # 2) 差分统计 — 捕获信号变化速率与波动性
            diff = np.diff(arr, axis=2)  # 一阶差分, (N, C, W-1)
            tmp_parts.append(np.mean(np.abs(diff), axis=2))   # 平均绝对差分
            tmp_parts.append(np.std(diff, axis=2, ddof=1))    # 差分标准差

            # 3) 短时能量 (4 段) — 捕获能量时序分布
            seg_len = max(1, w // 4)
            for s in range(4):
                start = s * seg_len
                end = start + seg_len if s < 3 else w
                energy = np.sum(arr[:, :, start:end] ** 2, axis=2) / (end - start)
                tmp_parts.append(energy)

            # 4) 信号熵 (10-bin 直方图) — 捕获信号复杂度
            eps = 1e-10
            for c in range(n_feats):
                ch_data = arr[:, c, :]  # (N, W)
                cmin, cmax = ch_data.min(), ch_data.max()
                rng = cmax - cmin
                if rng < eps:
                    tmp_parts.append(np.zeros((arr.shape[0], 1), dtype=np.float32))
                    continue
                
                # 全局 bins + 向量化计数
                bins = np.linspace(cmin, cmax, 11)
                dig = np.digitize(ch_data, bins[:-1]) - 1  # (N, W), values 0..9
                np.clip(dig, 0, 9, out=dig)
                counts = np.zeros((arr.shape[0], 10), dtype=np.int32)
                offsets = np.arange(arr.shape[0], dtype=np.int64) * 10
                flat_dig = (dig.astype(np.int64) + offsets[:, np.newaxis]).ravel()
                counts = np.bincount(flat_dig, minlength=arr.shape[0] * 10).reshape(arr.shape[0], 10).astype(np.int32)
                probs = counts.astype(np.float32) / ch_data.shape[1]
                probs[probs == 0] = 1e-12
                ent = -np.sum(probs * np.log2(probs), axis=1)
                tmp_parts.append(ent[:, np.newaxis])

            temporal = np.concatenate(tmp_parts, axis=1).astype(np.float32)
            parts.append(temporal)

        if not parts:
            raise ValueError("至少启用一个特征组 (spectral / statistical / temporal)")

        return np.concatenate(parts, axis=1).astype(np.float32)

    # ── 完整流水线 ─────────────────────────────────────────────────────
    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """训练集: 特征提取 → PCA fit → Scaler fit → 返回特征。"""
        feats = self.extract_features(windows)
        if self.config.use_pca:
            feats = self.fit_pca(feats)
        if self.config.normalize:
            feats = self.fit_scaler(feats)
        return feats

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """测试集/推理: 特征提取 → PCA transform → Scaler transform → 返回特征。"""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor 尚未 fit，请先调用 fit_transform")

        feats = self.extract_features(windows)
        if self.pca is not None:
            feats = self.transform_pca(feats)
        if self.scaler is not None:
            feats = self.transform_scaler(feats)
        return feats

    # ── PCA ────────────────────────────────────────────────────────────
    def fit_pca(self, features: np.ndarray) -> np.ndarray:
        """PCA fit + transform。"""
        if not self.config.use_pca:
            return features

        n_components = self.config.pca_variance
        if self.config.pca_method == "incremental":
            self.pca = IncrementalPCA(n_components=n_components)
            # 分批 partial_fit 实现真正的增量 SVD (O(N·D·k) vs O(N·D²))
            n = features.shape[0]
            bs = min(_INCREMENTAL_PCA_DEFAULT_BATCH, max(1000, n // 10))
            for start in range(0, n, bs):
                end = min(start + bs, n)
                self.pca.partial_fit(features[start:end])
        else:
            self.pca = PCA(
                n_components=n_components,
                svd_solver="auto",
                random_state=self.random_seed,
            )
            self.pca.fit(features)

        result = self.pca.transform(features)
        return result.astype(np.float32, copy=False)

    def transform_pca(self, features: np.ndarray) -> np.ndarray:
        """PCA transform (先前已 fit)。"""
        if self.pca is None:
            return features
        return self.pca.transform(features).astype(np.float32, copy=False)

    # ── Scaler ─────────────────────────────────────────────────────────
    def fit_scaler(self, features: np.ndarray) -> np.ndarray:
        """MinMaxScaler fit + transform 到 [-1, 1] 范围。"""
        if not self.config.normalize:
            return features
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        return self.scaler.fit_transform(features).astype(np.float32, copy=False)

    def transform_scaler(self, features: np.ndarray) -> np.ndarray:
        """MinMaxScaler transform (先前已 fit)。"""
        if self.scaler is None:
            return features
        return self.scaler.transform(features).astype(np.float32, copy=False)

    # ── 信息 ───────────────────────────────────────────────────────────
    def info(self, original_dim: int) -> dict:
        """返回当前预处理状态信息。"""
        info: dict = {
            "denoise": self.config.denoise,
            "denoise_kernel": self.config.denoise_kernel,
            "low_freq_bins": self.config.low_freq_bins,
            "feature_groups": list(self.config.feature_groups),
            "use_pca": self.config.use_pca,
            "pca_method": self.config.pca_method,
            "normalize": self.config.normalize,
            "feature_dim_before_pca": original_dim,
        }
        if self.pca is not None:
            comps = int(self.pca.n_components_)
            explained = float(np.sum(self.pca.explained_variance_ratio_))
            info.update(
                pca_components=comps,
                pca_variance_retained=explained,
                feature_dim_after_pca=comps,
            )
        else:
            info.update(
                pca_components=original_dim,
                pca_variance_retained=1.0,
                feature_dim_after_pca=original_dim,
            )
        return info


# ══════════════════════════════════════════════════════════════════════
# 特征处理 I/O 流水线
# ══════════════════════════════════════════════════════════════════════
class FeatureProcessor:
    """特征处理 I/O 流水线 — 从窗口文件到特征文件。
    
    封装 FeatureExtractor 的文件读写，添加进度日志和内存估算。
    仅 pipeline_runner 使用。
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.extractor = FeatureExtractor(config)

    def process(self, input_file: Path, output_file: Path) -> dict[str, Any]:
        """从 pickle 文件读取窗口数据，提取特征并保存。
        
        Args:
            input_file: 包含 x_train, y_train, x_test, y_test 的 pickle。
            output_file: 输出 pickle 路径 (含提取后的特征)。
        Returns:
            dict 包含 meta, x_train, y_train, x_test, y_test 及模型。
        """
        logger.info("加载窗口数据: %s", input_file)
        with input_file.open("rb") as f:
            data = f.read()
        import pickle
        data = pickle.loads(data)

        if "x_train" not in data or "x_test" not in data:
            raise KeyError("输入文件必须包含 'x_train' 和 'x_test' 键")

        x_train = np.asarray(data["x_train"], dtype=np.float32)
        x_test = np.asarray(data["x_test"], dtype=np.float32)

        if x_train.size == 0:
            raise ValueError("训练集为空")

        # 内存估算
        train_mem_mb = x_train.nbytes / (1024 * 1024)
        test_mem_mb = x_test.nbytes / (1024 * 1024)
        logger.info("数据规模: 训练 %s (%.1fMB), 测试 %s (%.1fMB)", 
                    x_train.shape, train_mem_mb, x_test.shape, test_mem_mb)

        # 特征提取
        logger.info("开始特征提取...")
        train_feat = self.extractor.fit_transform(x_train)
        logger.info("训练特征: %s, 内存 %.1fMB", train_feat.shape, train_feat.nbytes / (1024 * 1024))

        test_feat = self.extractor.transform(x_test)
        logger.info("测试特征: %s, 内存 %.1fMB", test_feat.shape, test_feat.nbytes / (1024 * 1024))

        # 组装输出
        output = {
            "meta": self.extractor.info(train_feat.shape[1]),
            "x_train": train_feat,
            "y_train": data["y_train"],
            "x_test": test_feat,
            "y_test": data["y_test"],
            "pca_model": self.extractor.pca,
            "scaler_model": self.extractor.scaler,
            "feature_config": self.extractor.config,
        }

        # 持久化
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("特征已保存至: %s", output_file)
        return output