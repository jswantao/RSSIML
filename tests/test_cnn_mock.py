from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.train_cnn_models import RSSICNNClassifier, predict_cnn_probabilities, predict_cnn_windows


class TestCnnMock(unittest.TestCase):
    """验证 1D CNN 模型的基础张量逻辑。"""

    def test_forward_and_prediction_shapes(self) -> None:
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        mock_windows = rng.normal(size=(12, 200, 52)).astype(np.float32)

        model = RSSICNNClassifier(input_channels=52, num_classes=5)
        device = torch.device("cpu")

        preds = predict_cnn_windows(model, mock_windows, device=device)
        probs = predict_cnn_probabilities(model, mock_windows, device=device)

        self.assertEqual(preds.shape, (12,))
        self.assertEqual(probs.shape, (12, 5))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
