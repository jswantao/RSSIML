# -*- coding: utf-8 -*-
"""RSSI 频域特征提取与降维模块。

提供从滑动窗口数据中提取频域特征（FFT幅值）、统计特征（均值、方差、偏度、峰度、过零率），
并支持 PCA 降维和 Min-Max 归一化。
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from scripts.config import PipelineConfig

# 定义公共 API
__all__ = [
    "FeatureConfig",
    "FeatureExtractor",
    "FeatureProcessor",
    "extract_frequency_domain_features",
    "process_features",
]


@dataclass(frozen=True)
class FeatureConfig:
    """特征提取配置。

    Attributes:
        low_freq_bins: 保留的低频 FFT 幅值频点数。
        use_pca: 是否启用 PCA 降维。
        pca_variance: PCA 保留的累积方差比例 (0.0 - 1.0)。
    """

    low_freq_bins: int = 16
    use_pca: bool = True
    pca_variance: float = 0.9019


class FeatureExtractor:
    """频域与统计特征提取器。

    负责从原始滑动窗口数据中提取混合特征，并管理 PCA 和 Scaling 的状态。

    Attributes:
        feat_config: 特征提取配置。
        pca: 已拟合的 PCA 模型（若启用）。
        scaler: 已拟合的 MinMaxScaler 模型。
    """

    def __init__(
        self,
        feat_config: Optional[FeatureConfig] = None,
        random_seed: int = 42,
    ) -> None:
        """初始化特征提取器。

        Args:
            feat_config: 特征配置。若为 None，使用默认配置。
            random_seed: 随机种子，用于 PCA 等随机算法的复现。
        """
        self.feat_config = feat_config or FeatureConfig()
        self.random_seed = random_seed
        
        # 初始化模型容器
        self.pca: Optional[PCA] = None
        # 归一化到 [-1, 1] 区间，有助于神经网络收敛
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def _compute_spectral_features(
        self, centered: np.ndarray
    ) -> np.ndarray:
        """计算频域低频幅值特征。

        对去均值后的信号进行实数快速傅里叶变换 (RFFT)，提取前 K 个低频分量的幅值。

        Args:
            centered: 去均值后的窗口数据，形状 (N, W, C)。

        Returns:
            频域特征矩阵，形状 (N, low_freq_bins * C)。
        """
        # RFFT 沿时间轴 (axis=1) 进行
        # 输出形状: (N, W//2 + 1, C)
        spectrum = np.fft.rfft(centered, axis=1)
        magnitude = np.abs(spectrum).astype(np.float32)

        max_available_bins = magnitude.shape[1]
        keep_bins = min(self.feat_config.low_freq_bins, max_available_bins)

        # 提取前 keep_bins 个频点
        # 形状: (N, keep_bins, C)
        low_freq_mag = magnitude[:, :keep_bins, :]
        
        n_samples = low_freq_mag.shape[0]
        n_channels = low_freq_mag.shape[2]
        
        # 展平最后两维: (N, keep_bins * C)
        return low_freq_mag.reshape(n_samples, -1)

    def _compute_statistical_features(
        self, centered: np.ndarray, window_size: int
    ) -> np.ndarray:
        """计算时域统计特征。

        包括：标准差、偏度、峰度、过零率 (ZCR)。

        Args:
            centered: 去均值后的窗口数据，形状 (N, W, C)。
            window_size: 窗口长度 W。

        Returns:
            统计特征矩阵，形状 (N, 4 * C)。
            顺序: [Std, Skew, Kurt, ZCR] 对于每个通道。
        """
        # 1. 标准差 (Standard Deviation)
        # ddof=1 表示样本标准差
        std_feat = np.std(centered, axis=1, ddof=1).astype(np.float32)

        # 2. 偏度 (Skewness)
        # bias=False 使用无偏估计
        skew_feat = skew(centered, axis=1, bias=False).astype(np.float32)

        # 3. 峰度 (Kurtosis)
        # scipy 的 kurtosis 默认计算 excess kurtosis (峰度 - 3)
        # bias=False 使用无偏估计
        kurt_feat = kurtosis(centered, axis=1, bias=False).astype(np.float32)

        # 4. 过零率 (Zero Crossing Rate, ZCR)
        # 计算相邻样本符号变化的次数
        # sign 变化: positive->negative or negative->positive
        # 注意: 0 值的处理。np.sign(0)=0。如果信号穿过0，sign会变化。
        signs = np.sign(centered)
        # 差分后非零表示符号变化
        sign_changes = np.diff(signs, axis=1) != 0
        # 统计每个通道的变化次数
        zcr_counts = np.count_nonzero(sign_changes, axis=1).astype(np.float32)
        # 归一化到 [0, 1] 区间
        zcr_feat = zcr_counts / max(1, window_size - 1)

        # 拼接: (N, C) + (N, C) + (N, C) + (N, C) -> (N, 4*C)
        return np.concatenate([std_feat, skew_feat, kurt_feat, zcr_feat], axis=1)

    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """从滑动窗口中提取混合特征。

        Args:
            windows: 三维数组，形状 (N, W, C)。

        Returns:
            二维特征数组，形状 (N, D_raw)。
            D_raw = (low_freq_bins * C) + (4 * C)。

        Raises:
            ValueError: 当输入维度错误或窗口过短时抛出。
        """
        if windows.ndim != 3:
            raise ValueError(f"输入必须是三维数组 (N, W, C)，实际形状: {windows.shape}")

        n_samples, window_size, n_channels = windows.shape
        
        if window_size < 2:
            raise ValueError(f"窗口长度至少为 2，实际为: {window_size}")

        # 确保 float32
        windows = np.asarray(windows, dtype=np.float32)
        
        # 去均值 (Detrending)
        # 沿时间轴 (axis=1) 减去均值
        mean_val = np.mean(windows, axis=1, keepdims=True)
        centered = windows - mean_val

        # 1. 频域特征
        spectral = self._compute_spectral_features(centered)
        
        # 2. 统计特征
        statistical = self._compute_statistical_features(centered, window_size)

        # 3. 拼接
        # 形状: (N, low_freq_bins*C + 4*C)
        return np.concatenate([spectral, statistical], axis=1)

    def fit_pca(self, features: np.ndarray) -> np.ndarray:
        """在训练集上拟合并应用 PCA 降维。

        Args:
            features: 原始特征矩阵，形状 (N_train, D_raw)。

        Returns:
            降维后的特征矩阵，形状 (N_train, D_pca)。
        """
        if not self.feat_config.use_pca:
            return features

        # 初始化 PCA
        self.pca = PCA(
            n_components=self.feat_config.pca_variance,
            svd_solver="full",  # 精确求解，适合中小规模数据
            random_state=self.random_seed,
        )
        
        transformed = self.pca.fit_transform(features)
        return transformed.astype(np.float32)

    def transform_pca(self, features: np.ndarray) -> np.ndarray:
        """应用已拟合的 PCA 模型进行变换。

        Args:
            features: 原始特征矩阵，形状 (N, D_raw)。

        Returns:
            降维后的特征矩阵，形状 (N, D_pca)。
        """
        if self.pca is None:
            # 如果未启用 PCA 或未拟合，直接返回
            return features
        
        transformed = self.pca.transform(features)
        return transformed.astype(np.float32)

    def fit_scaler(self, features: np.ndarray) -> np.ndarray:
        """在训练集上拟合并应用归一化。

        Args:
            features: 输入特征矩阵。

        Returns:
            归一化后的特征矩阵。
        """
        return self.scaler.fit_transform(features).astype(np.float32)

    def transform_scaler(self, features: np.ndarray) -> np.ndarray:
        """应用已拟合的缩放器进行归一化。

        Args:
            features: 输入特征矩阵。

        Returns:
            归一化后的特征矩阵。
        """
        return self.scaler.transform(features).astype(np.float32)

    def get_pca_info(self, original_dim: int) -> dict:
        """获取 PCA 降维的详细元信息。

        Args:
            original_dim: 降维前的特征维度。

        Returns:
            包含 PCA 配置和结果的字典。
        """
        if self.pca is not None:
            n_components = int(self.pca.n_components_)
            explained_var = float(np.sum(self.pca.explained_variance_ratio_))
            return {
                "use_pca": True,
                "pca_variance_target": self.feat_config.pca_variance,
                "pca_components": n_components,
                "pca_explained_variance": explained_var,
                "feature_dim_before_pca": original_dim,
                "feature_dim_after_pca": n_components,
            }
        else:
            return {
                "use_pca": False,
                "pca_variance_target": self.feat_config.pca_variance,
                "pca_components": original_dim,
                "pca_explained_variance": 1.0,
                "feature_dim_before_pca": original_dim,
                "feature_dim_after_pca": original_dim,
            }


class FeatureProcessor:
    """特征处理完整流水线。

    整合特征提取、PCA 降维和归一化步骤。
    严格遵循“在训练集拟合，在测试集变换”的原则，防止数据泄露。

    Attributes:
        config: 流水线全局配置。
        extractor: 特征提取器实例。
    """

    def __init__(
        self,
        feat_config: Optional[FeatureConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化特征处理器。

        Args:
            feat_config: 特征提取配置。
            config: 流水线全局配置。
        """
        self.config = config or PipelineConfig.from_root()
        self.extractor = FeatureExtractor(
            feat_config=feat_config, 
            random_seed=self.config.random_seed
        )

    def process(
        self,
        input_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict:
        """执行完整的特征处理流程。

        1. 加载窗口化数据。
        2. 提取原始特征。
        3. PCA 降维 (Fit on Train, Transform on Test)。
        4. 归一化 (Fit on Train, Transform on Test)。
        5. 保存结果。

        Args:
            input_file: 输入窗口数据文件 (.pkl)。若为 None，使用默认路径。
            output_file: 输出特征数据文件 (.pkl)。若为 None，使用默认路径。

        Returns:
            包含处理后数据和元信息的字典。

        Raises:
            FileNotFoundError: 当输入文件不存在时抛出。
        """
        input_path = input_file or (self.config.data_dir / "rssi_windowed.pkl")
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 加载数据
        with input_path.open("rb") as f:
            data = pickle.load(f)

        x_train_raw = np.asarray(data["x_train"], dtype=np.float32)
        x_test_raw = np.asarray(data["x_test"], dtype=np.float32)
        y_train = data["y_train"]
        y_test = data["y_test"]

        if x_train_raw.size == 0:
            raise ValueError("训练集数据为空，无法提取特征")

        # 1. 特征提取
        feat_train = self.extractor.extract_features(x_train_raw)
        feat_test = self.extractor.extract_features(x_test_raw)
        
        original_dim = feat_train.shape[1]

        # 2. PCA 降维
        train_reduced = self.extractor.fit_pca(feat_train)
        test_reduced = self.extractor.transform_pca(feat_test)

        # 3. 归一化
        x_train_norm = self.extractor.fit_scaler(train_reduced)
        x_test_norm = self.extractor.transform_scaler(test_reduced)

        # 构建元数据
        pca_info = self.extractor.get_pca_info(original_dim)
        meta = {
            **pca_info,
            "feature_mode": "fft_lowfreq_stats",
            "low_freq_bins": self.extractor.feat_config.low_freq_bins,
            "train_samples": int(x_train_norm.shape[0]),
            "test_samples": int(x_test_norm.shape[0]),
            "final_feature_dim": int(x_train_norm.shape[1]),
        }

        output = {
            "meta": meta,
            "x_train": x_train_norm,
            "y_train": y_train,
            "x_test": x_test_norm,
            "y_test": y_test,
            # 保存模型对象以便后续推理时使用
            "pca_model": self.extractor.pca,
            "scaler_model": self.extractor.scaler,
        }

        # 保存结果
        out_path = output_file or (self.config.data_dir / "rssi_processed.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_path.open("wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        return output


# =============================================================================
# 便捷函数
# =============================================================================

def extract_frequency_domain_features(
    windows: np.ndarray, low_freq_bins: int = 16
) -> np.ndarray:
    """便捷函数：仅提取频域与统计特征（不进行 PCA/归一化）。

    Args:
        windows: 三维滑窗数据 (N, W, C)。
        low_freq_bins: 低频幅值频点数。

    Returns:
        二维特征矩阵 (N, D)。
    """
    config = FeatureConfig(low_freq_bins=low_freq_bins, use_pca=False)
    extractor = FeatureExtractor(feat_config=config)
    return extractor.extract_features(windows)


def process_features(
    windowed_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    low_freq_bins: int = 16,
    use_pca: bool = True,
    pca_variance: float = 0.9019,
    config: Optional[PipelineConfig] = None,
) -> dict:
    """便捷函数：执行完整特征处理流水线。

    Args:
        windowed_file: 输入窗口文件路径。
        output_file: 输出文件路径。
        low_freq_bins: 低频幅值频点数。
        use_pca: 是否启用 PCA。
        pca_variance: PCA 保留方差比例。
        config: 流水线配置。

    Returns:
        包含处理后数据和元信息的字典。
    """
    feat_config = FeatureConfig(
        low_freq_bins=low_freq_bins,
        use_pca=use_pca,
        pca_variance=pca_variance,
    )
    processor = FeatureProcessor(feat_config=feat_config, config=config)
    return processor.process(input_file=windowed_file, output_file=output_file)


# =============================================================================
# 命令行入口
# =============================================================================

def main() -> None:
    """主函数：解析命令行参数并执行特征处理。"""
    parser = argparse.ArgumentParser(description="RSSI 频域特征提取与降维")
    parser.add_argument(
        "--low-freq-bins", type=int, default=16, help="低频幅值频点数，默认 16"
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.9019,
        help="PCA 方差保留比例，默认 0.9019",
    )
    parser.add_argument(
        "--disable-pca", action="store_true", help="禁用 PCA 降维"
    )
    parser.add_argument(
        "--input", type=Path, help="输入窗口文件路径（可选）"
    )
    parser.add_argument(
        "--output", type=Path, help="输出特征文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    feat_config = FeatureConfig(
        low_freq_bins=args.low_freq_bins,
        pca_variance=args.pca_variance,
        use_pca=not args.disable_pca,
    )

    processor = FeatureProcessor(feat_config=feat_config)
    
    try:
        result = processor.process(input_file=args.input, output_file=args.output)
        
        print("✅ 特征处理完成")
        meta = result["meta"]
        print(f"  特征模式: {meta['feature_mode']}")
        print(f"  PCA: {'启用' if meta['use_pca'] else '禁用'}")
        if meta["use_pca"]:
            print(f"    - 降维前: {meta['feature_dim_before_pca']} 维")
            print(f"    - 降维后: {meta['feature_dim_after_pca']} 维")
            print(f"    - 解释方差: {meta['pca_explained_variance']:.4f}")
        print(f"  最终特征维度: {meta['final_feature_dim']}")
        print(f"  训练样本: {meta['train_samples']}")
        print(f"  测试样本: {meta['test_samples']}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()