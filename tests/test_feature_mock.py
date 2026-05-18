from __future__ import annotations

import unittest

import numpy as np

from scripts.process_features_pca_norm import (
    FeatureExtractor,
    PreprocessConfig,
)


class TestFeatureExtractor(unittest.TestCase):
    """验证特征提取器的基础行为。"""

    def test_extract_features_shape(self) -> None:
        """测试频域+统计+时域特征提取的输出维度。"""
        rng = np.random.default_rng(7)
        # (N=6, channels=52, window_size=200)
        windows = rng.normal(size=(6, 52, 200)).astype(np.float32)

        cfg = PreprocessConfig(
            use_pca=False, normalize=False,
            low_freq_bins=16,
            feature_groups=("spectral", "statistical", "temporal"),
        )
        fe = FeatureExtractor(cfg)
        features = fe.extract_features(windows)

        self.assertEqual(features.shape[0], 6)
        self.assertTrue(features.shape[1] > 0)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(features.dtype, np.float32)

    def test_spectral_only(self) -> None:
        """仅频域特征: 16 bins × 52 channels = 832。"""
        rng = np.random.default_rng(11)
        windows = rng.normal(size=(4, 52, 200)).astype(np.float32)

        cfg = PreprocessConfig(
            use_pca=False, normalize=False,
            low_freq_bins=16,
            feature_groups=("spectral",),
        )
        fe = FeatureExtractor(cfg)
        features = fe.extract_features(windows)

        self.assertEqual(features.shape, (4, 16 * 52))

    def test_fit_transform_and_transform(self) -> None:
        """训练集 fit_transform + 测试集 transform 维度一致。"""
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(20, 10, 100)).astype(np.float32)
        x_test = rng.normal(size=(5, 10, 100)).astype(np.float32)

        cfg = PreprocessConfig(
            use_pca=True, pca_variance=0.95,
            pca_method="full",
            normalize=True,
            low_freq_bins=8,
            feature_groups=("spectral", "statistical"),
        )
        fe = FeatureExtractor(cfg)

        train_feat = fe.fit_transform(x_train)
        test_feat = fe.transform(x_test)

        self.assertEqual(train_feat.shape[0], 20)
        self.assertEqual(test_feat.shape[0], 5)
        # PCA 后维度应一致
        self.assertEqual(train_feat.shape[1], test_feat.shape[1])

    def test_denoise(self) -> None:
        """降噪不改变数组形状。"""
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(3, 5, 50)).astype(np.float32)

        for method in ("moving_avg", "median", "lowpass"):
            cfg = PreprocessConfig(
                use_pca=False, normalize=False,
                denoise=method, denoise_kernel=5,
                feature_groups=("spectral",),
            )
            fe = FeatureExtractor(cfg)
            denoised = fe.denoise(windows)
            self.assertEqual(denoised.shape, windows.shape, f"denoise={method}")

    def test_config_validation(self) -> None:
        """配置校验应拒绝非法参数。"""
        with self.assertRaises(ValueError):
            PreprocessConfig(denoise="invalid_method")

        with self.assertRaises(ValueError):
            PreprocessConfig(denoise_kernel=4)  # 必须为奇数

        with self.assertRaises(ValueError):
            PreprocessConfig(pca_variance=1.5)  # 必须在 (0, 1)

        with self.assertRaises(ValueError):
            PreprocessConfig(feature_groups=("invalid",))


if __name__ == "__main__":
    unittest.main()
