from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.process_features_pca_norm as feature_module


class DummyPca:
    """用于验证 PCA 调用顺序的轻量替身。"""

    def __init__(self, n_components=None, svd_solver=None, random_state=None) -> None:
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.random_state = random_state
        self.fit_shape: tuple[int, int] | None = None
        self.transform_shape: tuple[int, int] | None = None
        self.n_components_ = 12
        self.explained_variance_ratio_ = np.array([0.5], dtype=np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit_shape = tuple(int(v) for v in x.shape)
        return np.asarray(x, dtype=np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.transform_shape = tuple(int(v) for v in x.shape)
        return np.asarray(x, dtype=np.float32)


class TestFeatureMock(unittest.TestCase):
    """验证频域特征与 PCA 流程的基础行为。"""

    def test_frequency_features_shape(self) -> None:
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(6, 200, 52)).astype(np.float32)

        features = feature_module.extract_frequency_domain_features(windows, low_freq_bins=12)

        self.assertEqual(features.shape, (6, 12 * 52 + 4 * 52))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_pca_uses_train_and_test_separately(self) -> None:
        rng = np.random.default_rng(11)
        x_train = rng.normal(size=(4, 200, 52)).astype(np.float32)
        x_test = rng.normal(size=(2, 200, 52)).astype(np.float32)

        payload = {
            "x_train": x_train,
            "y_train": np.array(["A", "B", "A", "B"], dtype=object),
            "x_test": x_test,
            "y_test": np.array(["A", "B"], dtype=object),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "windowed.pkl"
            output_path = Path(temp_dir) / "processed.pkl"
            with input_path.open("wb") as f:
                pickle.dump(payload, f)

            with patch.object(feature_module, "PCA", DummyPca):
                result = feature_module.process_features(
                    windowed_file=input_path,
                    output_file=output_path,
                    use_pca=True,
                    low_freq_bins=12,
                )

        self.assertEqual(result["meta"]["feature_mode"], "fft_lowfreq_stats")
        self.assertEqual(result["meta"]["low_freq_bins"], 12)
        self.assertEqual(result["x_train"].shape[0], 4)
        self.assertEqual(result["x_test"].shape[0], 2)
        self.assertEqual(result["x_train"].shape[1], result["x_test"].shape[1])

        pca_obj = result["pca"]
        self.assertIsInstance(pca_obj, DummyPca)
        self.assertEqual(pca_obj.fit_shape[0], 4)
        self.assertEqual(pca_obj.transform_shape[0], 2)


if __name__ == "__main__":
    unittest.main()