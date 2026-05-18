from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from scripts.models.cnn.models import CNNConfig, RSSICNNBinaryClassifier
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch 不可用")
class TestCnnMock(unittest.TestCase):
    """验证 1D CNN 模型的基础张量逻辑。"""

    def test_forward_shape(self) -> None:
        """测试前向传播输出形状。"""
        torch.manual_seed(7)
        config = CNNConfig(conv_channels=(32, 64), hidden_units=64)
        model = RSSICNNBinaryClassifier(
            input_channels=52, num_classes=1, config=config,
        )
        model.eval()

        # (batch=12, channels=52, window_size=200)
        x = torch.randn(12, 52, 200)
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape, (12, 1))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_binary_prediction(self) -> None:
        """测试 sigmoid 概率输出范围。"""
        torch.manual_seed(42)
        config = CNNConfig(conv_channels=(32,), hidden_units=32)
        model = RSSICNNBinaryClassifier(
            input_channels=10, num_classes=1, config=config,
        )
        model.eval()

        x = torch.randn(8, 10, 100)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)

        self.assertEqual(probs.shape, (8, 1))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))


if __name__ == "__main__":
    unittest.main()
