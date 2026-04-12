"""RSSI 频域特征提取与降维模块。

提供从滑动窗口数据中提取频域特征、PCA 降维和归一化功能。
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from scripts.config import PipelineConfig


@dataclass(frozen=True)
class FeatureConfig:
    """特征提取配置。
    
    Attributes:
        low_freq_bins: 保留的低频幅值频点数。
        use_pca: 是否使用 PCA 降维。
        pca_variance: PCA 保留的方差比例。
    """
    low_freq_bins: int = 16
    use_pca: bool = True
    pca_variance: float = 0.9019


class FeatureExtractor:
    """频域特征提取器。
    
    从滑动窗口数据中提取频域和统计特征，支持 PCA 降维和归一化。
    
    Attributes:
        config: 流水线配置对象。
        feat_config: 特征提取配置。
        pca: PCA 模型（仅当启用时）。
        scaler: 归一化缩放器。
    """
    
    def __init__(self, feat_config: Optional[FeatureConfig] = None) -> None:
        """初始化特征提取器。
        
        Args:
            feat_config: 特征提取配置，None 时使用默认配置。
        """
        self.config = PipelineConfig.from_root()
        self.feat_config = feat_config or FeatureConfig()
        self.pca: Optional[PCA] = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def _compute_spectral_features(
        self, windows: np.ndarray, centered: np.ndarray
    ) -> np.ndarray:
        """计算频域低频幅值特征。
        
        Args:
            windows: 原始窗口数据，形状 (n_samples, window_size, n_channels)。
            centered: 去均值后的窗口数据。
            
        Returns:
            频域特征矩阵，形状 (n_samples, low_freq_bins * n_channels)。
        """
        spectrum = np.fft.rfft(centered, axis=1)
        magnitude = np.abs(spectrum).astype(np.float32)
        
        max_bins = magnitude.shape[1]
        keep_bins = min(self.feat_config.low_freq_bins, max_bins)
        
        n_samples, _, n_channels = windows.shape
        return magnitude[:, :keep_bins, :].reshape(n_samples, -1)
    
    def _compute_statistical_features(
        self, centered: np.ndarray, window_size: int
    ) -> np.ndarray:
        """计算统计特征：标准差、偏度、峰度、过零率。
        
        Args:
            centered: 去均值后的窗口数据。
            window_size: 窗口长度。
            
        Returns:
            统计特征矩阵，形状 (n_samples, 4 * n_channels)。
        """
        eps = np.float32(1e-8)
        
        # 标准差
        std_feat = np.std(centered, axis=1)
        std_safe = np.maximum(std_feat, eps)
        
        # 偏度和峰度
        skew_feat = np.mean(centered ** 3, axis=1) / (std_safe ** 3)
        kurt_feat = np.mean(centered ** 4, axis=1) / (std_safe ** 4) - 3.0
        
        # 过零率
        sign_changes = np.diff(np.signbit(centered), axis=1)
        zcr_feat = np.count_nonzero(sign_changes, axis=1).astype(np.float32) / (window_size - 1)
        
        return np.concatenate([std_feat, skew_feat, kurt_feat, zcr_feat], axis=1).astype(np.float32)
    
    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """从滑动窗口中提取频域与统计特征。
        
        Args:
            windows: 三维数组，形状 (n_samples, window_size, n_channels)。
            
        Returns:
            二维特征数组，形状 (n_samples, n_features)。
            
        Raises:
            ValueError: 当输入数据维度不为 3 或窗口长度不足时抛出。
        """
        if windows.ndim != 3:
            raise ValueError(f"输入维度应为 3，实际为 {windows.shape}")
        
        n_samples, window_size, n_channels = windows.shape
        if window_size < 2:
            raise ValueError(f"窗口长度过短: {window_size}")
        
        windows = np.asarray(windows, dtype=np.float32)
        centered = windows - np.mean(windows, axis=1, keepdims=True)
        
        # 提取频域和统计特征
        spectral = self._compute_spectral_features(windows, centered)
        statistical = self._compute_statistical_features(centered, window_size)
        
        return np.concatenate([spectral, statistical], axis=1)
    
    def fit_pca(self, features: np.ndarray) -> np.ndarray:
        """拟合并应用 PCA 降维。
        
        Args:
            features: 原始特征矩阵。
            
        Returns:
            降维后的特征矩阵。
        """
        if not self.feat_config.use_pca:
            return features
        
        self.pca = PCA(
            n_components=self.feat_config.pca_variance,
            svd_solver="full",
            random_state=self.config.random_seed,
        )
        return self.pca.fit_transform(features).astype(np.float32)
    
    def transform_pca(self, features: np.ndarray) -> np.ndarray:
        """应用已拟合的 PCA 模型进行降维。
        
        Args:
            features: 原始特征矩阵。
            
        Returns:
            降维后的特征矩阵。
        """
        if self.pca is None:
            return features
        return self.pca.transform(features).astype(np.float32)
    
    def fit_scaler(self, features: np.ndarray) -> np.ndarray:
        """拟合并应用归一化。
        
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
        """获取 PCA 降维信息。
        
        Args:
            original_dim: 降维前的特征维度。
            
        Returns:
            包含 PCA 配置和结果的字典。
        """
        if self.pca is not None:
            return {
                "use_pca": True,
                "pca_variance_target": self.feat_config.pca_variance,
                "pca_components": int(self.pca.n_components_),
                "pca_explained_variance": float(np.sum(self.pca.explained_variance_ratio_)),
                "feature_dim_before_pca": original_dim,
                "feature_dim_after_pca": int(self.pca.n_components_),
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
    """特征处理流水线。
    
    整合数据加载、特征提取、降维和归一化的完整流程。
    
    Attributes:
        config: 流水线配置对象。
        extractor: 特征提取器。
    """
    
    def __init__(self, feat_config: Optional[FeatureConfig] = None) -> None:
        """初始化特征处理器。
        
        Args:
            feat_config: 特征提取配置。
        """
        self.config = PipelineConfig.from_root()
        self.extractor = FeatureExtractor(feat_config)
    
    def process(
        self,
        input_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict:
        """执行完整的特征处理流程。
        
        Args:
            input_file: 输入滑窗数据文件路径，None 时使用默认路径。
            output_file: 输出文件路径，None 时使用默认路径。
            
        Returns:
            包含处理后数据和元信息的字典。
        """
        # 加载数据
        input_path = input_file or self.config.data_dir / "rssi_windowed.pkl"
        with input_path.open("rb") as f:
            data = pickle.load(f)
        
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        
        # 提取特征
        feat_train = self.extractor.extract_features(x_train)
        feat_test = self.extractor.extract_features(x_test)
        original_dim = feat_train.shape[1]
        
        # PCA 降维
        train_reduced = self.extractor.fit_pca(feat_train)
        test_reduced = self.extractor.transform_pca(feat_test)
        
        # 归一化
        x_train_norm = self.extractor.fit_scaler(train_reduced)
        x_test_norm = self.extractor.transform_scaler(test_reduced)
        
        # 构建输出
        pca_info = self.extractor.get_pca_info(original_dim)
        output = {
            "meta": {
                **pca_info,
                "feature_mode": "fft_lowfreq_stats",
                "low_freq_bins": self.extractor.feat_config.low_freq_bins,
                "train_samples": x_train_norm.shape[0],
                "test_samples": x_test_norm.shape[0],
            },
            "x_train": x_train_norm,
            "y_train": data["y_train"],
            "x_test": x_test_norm,
            "y_test": data["y_test"],
            "pca": self.extractor.pca,
            "scaler": self.extractor.scaler,
        }
        
        # 保存结果
        out_path = output_file or self.config.data_dir / "rssi_processed.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(output, f)
        
        return output


def main() -> None:
    """主函数：解析命令行参数并执行特征处理。"""
    parser = argparse.ArgumentParser(description="RSSI 频域特征提取与降维")
    parser.add_argument("--low-freq-bins", type=int, default=16, help="低频幅值频点数，默认 16")
    parser.add_argument("--pca-variance", type=float, default=0.9019, help="PCA 方差保留比例，默认 0.9019")
    parser.add_argument("--disable-pca", action="store_true", help="禁用 PCA 降维")
    args = parser.parse_args()
    
    # 创建配置并执行处理
    feat_config = FeatureConfig(
        low_freq_bins=args.low_freq_bins,
        pca_variance=args.pca_variance,
        use_pca=not args.disable_pca,
    )
    
    processor = FeatureProcessor(feat_config)
    result = processor.process()
    
    # 输出结果摘要
    print("特征处理完成")
    meta = result["meta"]
    print(f"  特征模式: {meta['feature_mode']}")
    print(f"  PCA: {'启用' if meta['use_pca'] else '禁用'}")
    if meta['use_pca']:
        print(f"    - 降维前: {meta['feature_dim_before_pca']} 维")
        print(f"    - 降维后: {meta['feature_dim_after_pca']} 维")
        print(f"    - 解释方差: {meta['pca_explained_variance']:.4f}")
    print(f"  训练样本: {meta['train_samples']}")
    print(f"  测试样本: {meta['test_samples']}")


if __name__ == "__main__":
    main()