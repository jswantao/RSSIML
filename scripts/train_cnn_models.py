# -*- coding: utf-8 -*-
"""RSSI 1D CNN 模型训练模块。

提供基于卷积神经网络的 RSSI 时序数据分类和开集识别功能。
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from scripts.config import PipelineConfig
from scripts.train_and_validate_models import MetricsCalculator

# 定义公共 API
__all__ = [
    "RSSICNNClassifier",
    "CNNConfig",
    "TrainingConfig",
    "CNNCheckpoint",
    "CNNTrainer",
    "CNNInference",
    "train_cnn_classification",
    "train_cnn_identification",
    "load_cnn_checkpoint",
    "predict_cnn_windows",
    "predict_cnn_probabilities",
    "predict_cnn_embeddings",
]


def _require_torch() -> None:
    """检查 PyTorch 是否已安装。

    Raises:
        ImportError: 当 PyTorch 未安装时抛出。
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for CNN training. Please install it via 'pip install torch'."
        )


# =============================================================================
# 模型定义
# =============================================================================


class RSSICNNClassifier(nn.Module):
    """RSSI 1D CNN 分类器。

    Attributes:
        conv_layers: 卷积层序列。
        fc_layers: 全连接层序列。
        embedding_layer: 用于提取嵌入的全连接层（倒数第二层）。
    """

    def __init__(
        self, input_channels: int, num_classes: int, config: CNNConfig
    ):
        super(RSSICNNClassifier, self).__init__()

        layers = []
        in_ch = input_channels
        for out_ch in config.conv_channels:
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=config.kernel_size,
                        padding=config.padding,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rates[0]),
                ]
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # 动态计算全连接层的输入维度
        dummy_input = torch.randn(1, input_channels, config.window_size)
        conv_out = self.conv_layers(dummy_input)
        fc_input_dim = conv_out.view(conv_out.size(0), -1).shape[1]

        # 定义全连接层
        self.fc_hidden = nn.Sequential(
            nn.Linear(fc_input_dim, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout_rates[1]),
        )
        
        self.classifier = nn.Linear(config.hidden_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, channels, sequence_length)。

        Returns:
            输出 logits，形状 (batch_size, num_classes)。
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_hidden(x)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征嵌入（全连接隐藏层输出）。

        Args:
            x: 输入张量，形状 (batch_size, channels, sequence_length)。

        Returns:
            嵌入向量，形状 (batch_size, hidden_units)。
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_hidden(x)


class LegacyRSSICNNClassifier(nn.Module):
    """兼容旧版本检查点的 CNN 分类器。"""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        config: Dict[str, Any],
        window_size: int,
    ) -> None:
        super().__init__()

        conv_channels = list(config.get("conv_channels", (64, 128, 256)))
        kernel_sizes = list(config.get("kernel_sizes", (5, 3, 3)))
        dropout_rates = list(config.get("dropout_rates", (0.3, 0.2)))
        fc_units = int(config.get("fc_units", 128))

        features_layers: List[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            features_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout_rates[0]),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*features_layers)

        # 旧模型在分类头前会进行全局池化，将通道维压成固定长度向量。
        pooled_dim = int(conv_channels[-1]) if conv_channels else 256
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(pooled_dim, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        return x.view(x.size(0), -1)


# =============================================================================
# 配置数据类
# =============================================================================


@dataclass(frozen=True)
class CNNConfig:
    """CNN 模型配置。

    Attributes:
        conv_channels: 卷积层输出通道数列表。
        kernel_size: 卷积核大小。
        padding: 填充大小。
        hidden_units: 全连接层隐藏单元数。
        dropout_rates: Dropout 概率列表 [conv_layer_dropout, fc_layer_dropout]。
        window_size: 输入窗口大小（用于动态计算全连接层维度）。
    """

    conv_channels: Tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 7
    padding: int = 3
    hidden_units: int = 256
    dropout_rates: Tuple[float, float] = (0.2, 0.5)
    window_size: int = 200

    def __post_init__(self) -> None:
        """验证配置参数的有效性。

        Raises:
            ValueError: 当参数非法时抛出。
        """
        if len(self.conv_channels) < 1:
            raise ValueError(
                f"卷积层数必须大于 0，实际为 {len(self.conv_channels)}"
            )
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"卷积核大小必须为大于 0 的奇数，实际为 {self.kernel_size}"
            )
        if self.hidden_units < 1:
            raise ValueError(
                f"隐藏单元数必须大于 0，实际为 {self.hidden_units}"
            )
        if len(self.dropout_rates) != 2:
            raise ValueError(
                f"dropout_rates 必须包含 2 个元素，实际为 {len(self.dropout_rates)}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """训练配置。

    Attributes:
        epochs: 最大训练轮数。
        batch_size: 批大小。
        learning_rate: 学习率。
        weight_decay: 权重衰减。
        val_ratio: 验证集比例。
        early_stop_patience: 早停耐心轮数。
        random_seed: 随机种子。
    """

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    early_stop_patience: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        """验证配置参数的有效性。

        Raises:
            ValueError: 当参数非法时抛出。
        """
        if not 0 < self.val_ratio < 0.5:
            raise ValueError(
                f"验证集比例必须在 (0, 0.5) 之间，实际为 {self.val_ratio}"
            )
        if self.epochs < 1:
            raise ValueError(f"训练轮数必须大于 0，实际为 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"批大小必须大于 0，实际为 {self.batch_size}")


# =============================================================================
# 检查点与数据加载器
# =============================================================================


@dataclass
class CNNCheckpoint:
    """CNN 模型检查点。

    Attributes:
        state_dict: 模型状态字典。
        config: 模型配置。
        input_channels: 输入通道数。
        num_classes: 分类数。
        label_encoder: 标签编码器。
    """

    state_dict: Dict[str, Any]
    config: CNNConfig
    input_channels: int
    num_classes: int
    label_encoder: LabelEncoder
    centroids: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    task: str = "classification"


class DataLoaderFactory:
    """数据加载器工厂。

    Attributes:
        config: 训练配置。
        device: 计算设备。
    """

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        """初始化数据加载器工厂。

        Args:
            config: 训练配置。
            device: 计算设备。
        """
        self.config = config
        self.device = device

    def create(
        self,
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器。

        Args:
            x_fit: 训练特征 (N, W, C)。
            y_fit: 训练标签。
            x_val: 验证特征 (N, W, C)。
            y_val: 验证标签。

        Returns:
            (train_loader, val_loader) 元组。
        """
        # 转换为 (N, C, W) 以符合 Conv1d 输入要求
        x_fit_tensor = (
            torch.tensor(x_fit, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        )
        y_fit_tensor = torch.tensor(y_fit, dtype=torch.long).to(self.device)

        x_val_tensor = (
            torch.tensor(x_val, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        )
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(x_fit_tensor, y_fit_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader


# =============================================================================
# CNN 训练器
# =============================================================================


class CNNTrainer:
    """CNN 模型训练器。

    支持分类和开集识别两种任务的训练。

    Attributes:
        pipeline_config: 流水线配置。
        model_config: CNN 模型配置。
        train_config: 训练配置。
        device: 计算设备。
    """

    def __init__(
        self,
        model_config: Optional[CNNConfig] = None,
        train_config: Optional[TrainingConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化训练器。

        Args:
            model_config: CNN 模型配置。
            train_config: 训练配置。
            config: 流水线配置。
        """
        _require_torch()
        self.pipeline_config = config or PipelineConfig.from_root()
        self.model_config = model_config or CNNConfig()
        self.train_config = train_config or TrainingConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

    def _load_windowed_data(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载滑窗数据。

        Args:
            file_path: 数据文件路径，None 时使用默认路径。

        Returns:
            数据字典。

        Raises:
            FileNotFoundError: 当数据文件不存在时抛出。
        """
        path = file_path or (self.pipeline_config.data_dir / "rssi_windowed.pkl")
        if not path.exists():
            raise FileNotFoundError(f"滑窗数据文件不存在: {path}")

        with path.open("rb") as f:
            return pickle.load(f)

    def _prepare_data(
        self,
        data: Dict[str, Any],
        task: Literal["classification", "identification"],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """准备训练和测试数据。

        Args:
            data: 加载的数据字典。
            task: 任务类型。

        Returns:
            (x_train, y_train_enc, x_test, y_test_enc, encoder) 元组。
        """
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train_raw = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test_raw = np.asarray(data["y_test"], dtype=object)

        encoder = LabelEncoder()
        y_train_enc = np.asarray(encoder.fit_transform(y_train_raw), dtype=np.int64)

        if task == "classification":
            # 分类任务：y_test 也需要编码
            try:
                y_test_enc = np.asarray(
                    encoder.transform(y_test_raw), dtype=np.int64
                )
            except ValueError as e:
                raise ValueError(
                    f"测试集包含未知标签: {e}. 请检查数据划分。"
                ) from e
        else:
            # 识别任务：y_test_enc 保持原始标签值，后续处理
            y_test_enc = y_test_raw

        return x_train, y_train_enc, x_test, y_test_enc, encoder

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """执行一个训练或验证周期。

        Args:
            model: 模型。
            loader: 数据加载器。
            criterion: 损失函数。
            optimizer: 优化器，None 表示验证模式。

        Returns:
            (loss, targets, predictions) 元组。
        """
        is_training = optimizer is not None
        model.train(is_training)
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in loader:
            # 数据已经在 DataLoaderFactory 中移动到 device
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            targets = batch_y.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

        avg_loss = total_loss / len(loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return avg_loss, all_targets, all_preds

    def train_classification(
        self,
        data_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """训练 CNN 分类模型。

        Args:
            data_file: 数据文件路径。

        Returns:
            训练结果字典。
        """
        # 加载和准备数据
        data = self._load_windowed_data(data_file)
        x_train, y_train, x_test, y_test, encoder = self._prepare_data(
            data, "classification"
        )

        # 划分训练验证集
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.train_config.val_ratio,
            random_state=self.train_config.random_seed,
            stratify=y_train,
        )

        # 创建数据加载器
        factory = DataLoaderFactory(self.train_config, self.device)
        train_loader, val_loader = factory.create(x_fit, y_fit, x_val, y_val)

        # 构建模型
        input_channels = x_train.shape[2]
        num_classes = len(encoder.classes_)
        model = RSSICNNClassifier(input_channels, num_classes, self.model_config).to(
            self.device
        )

        # 训练
        history = self._train_model(model, train_loader, val_loader)

        # 测试评估
        test_preds = self._predict(model, x_test)
        test_preds_labels = encoder.inverse_transform(test_preds)
        y_test_labels = encoder.inverse_transform(y_test)
        
        # 使用 MetricsCalculator 保持一致性
        test_metrics_obj = MetricsCalculator.calculate(y_test_labels, test_preds_labels)
        test_metrics = test_metrics_obj.to_dict()

        # 保存检查点
        checkpoint = CNNCheckpoint(
            state_dict=model.state_dict(),
            config=self.model_config,
            input_channels=input_channels,
            num_classes=num_classes,
            label_encoder=encoder,
            task="classification",
        )
        checkpoint_path = self._save_checkpoint(checkpoint, "cnn_classification.pt")

        return self._build_result(
            "classification",
            test_metrics,
            history,
            encoder,
            checkpoint_path,
        )

    def train_identification(
        self,
        data_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """训练 CNN 开集识别模型。

        Args:
            data_file: 数据文件路径。

        Returns:
            训练结果字典。
        """
        # 加载和准备数据
        data = self._load_windowed_data(data_file)
        x_train, y_train, x_test, y_test_raw, encoder = self._prepare_data(
            data, "identification"
        )

        # 划分训练验证集
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.train_config.val_ratio,
            random_state=self.train_config.random_seed,
            stratify=y_train,
        )

        # 创建数据加载器
        factory = DataLoaderFactory(self.train_config, self.device)
        train_loader, val_loader = factory.create(x_fit, y_fit, x_val, y_val)

        # 构建模型
        input_channels = x_train.shape[2]
        num_classes = len(encoder.classes_)
        model = RSSICNNClassifier(input_channels, num_classes, self.model_config).to(
            self.device
        )

        # 训练基础分类器
        history = self._train_model(model, train_loader, val_loader)

        # 计算训练集上的嵌入和质心
        train_embeddings = self._extract_embeddings(model, x_train)
        unique_labels = np.unique(y_train)
        centroids = self._compute_centroids(train_embeddings, y_train, unique_labels)

        # 计算距离阈值
        threshold = self._compute_threshold(
            train_embeddings, y_train, centroids, unique_labels
        )

        # 评估识别模型
        y_pred_open, y_true_open = self._predict_identification_with_embeddings(
            model, x_test, y_test_raw, centroids, threshold, encoder
        )
        
        test_metrics_obj = MetricsCalculator.calculate(
            y_true_open, y_pred_open, average="binary", pos_label="known"
        )
        test_metrics = test_metrics_obj.to_dict()

        # 保存检查点
        checkpoint = CNNCheckpoint(
            state_dict=model.state_dict(),
            config=self.model_config,
            input_channels=input_channels,
            num_classes=num_classes,
            label_encoder=encoder,
            centroids=centroids,
            threshold=threshold,
            task="identification",
        )
        checkpoint_path = self._save_checkpoint(checkpoint, "cnn_identification.pt")

        extra_info = {
            "threshold": float(threshold),
            "centroid_labels": encoder.inverse_transform(unique_labels).tolist(),
        }

        return self._build_result(
            "identification",
            test_metrics,
            history,
            encoder,
            checkpoint_path,
            extra=extra_info,
        )

    @staticmethod
    def _compute_centroids(
        embeddings: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray
    ) -> np.ndarray:
        """计算每个类的质心。

        Args:
            embeddings: 嵌入向量。
            labels: 标签。
            unique_labels: 唯一标签。

        Returns:
            质心矩阵。
        """
        n_classes = len(unique_labels)
        dim = embeddings.shape[1]
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        counts = np.zeros(n_classes)
        
        for i, emb in enumerate(embeddings):
            lbl = labels[i]
            idx = label_to_idx[lbl]
            centroids[idx] += emb
            counts[idx] += 1
            
        # 避免除零
        counts[counts == 0] = 1
        centroids /= counts[:, np.newaxis]
        
        return centroids

    def _compute_threshold(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        unique_labels: np.ndarray,
    ) -> float:
        """计算开集识别的距离阈值。

        Args:
            embeddings: 训练嵌入。
            labels: 训练标签。
            centroids: 质心。
            unique_labels: 唯一标签。

        Returns:
            距离阈值。
        """
        distances = []
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        
        for i, emb in enumerate(embeddings):
            lbl = labels[i]
            centroid = centroids[label_to_idx[lbl]]
            dist = np.linalg.norm(emb - centroid)
            distances.append(dist)
            
        if not distances:
            return 0.0
            
        return float(np.quantile(distances, 0.95))

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict[str, Any]]:
        """训练模型。

        Args:
            model: 模型。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。

        Returns:
            训练历史记录列表。
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )

        best_state = None
        best_acc = -1.0
        no_improve = 0
        history = []

        for epoch in range(self.train_config.epochs):
            # 训练
            train_loss, train_targets, train_preds = self._run_epoch(
                model, train_loader, criterion, optimizer
            )
            train_metrics_obj = MetricsCalculator.calculate(train_targets, train_preds)
            train_metrics = train_metrics_obj.to_dict()
            train_metrics["loss"] = train_loss

            # 验证
            val_loss, val_targets, val_preds = self._run_epoch(
                model, val_loader, criterion, None
            )
            val_metrics_obj = MetricsCalculator.calculate(val_targets, val_preds)
            val_metrics = val_metrics_obj.to_dict()
            val_metrics["loss"] = val_loss

            scheduler.step(val_metrics["accuracy"])

            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            # 早停
            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.train_config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳权重
        if best_state is not None:
            model.load_state_dict(best_state)

        return history

    def _predict(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """模型预测。

        Args:
            model: 训练好的模型。
            x: 输入特征 (N, W, C)。

        Returns:
            预测标签数组。
        """
        model.eval()
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        )
        loader = DataLoader(
            dataset, batch_size=self.train_config.batch_size, shuffle=False
        )

        all_preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds)

    def _extract_embeddings(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """提取特征嵌入。

        Args:
            model: 训练好的模型。
            x: 输入特征 (N, W, C)。

        Returns:
            嵌入向量数组。
        """
        model.eval()
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        )
        loader = DataLoader(
            dataset, batch_size=self.train_config.batch_size, shuffle=False
        )

        embeddings_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                embeddings = model.get_embeddings(batch_x)
                embeddings_list.append(embeddings.cpu().numpy())

        return np.concatenate(embeddings_list)

    def _predict_identification_with_embeddings(
        self,
        model: nn.Module,
        x_test: np.ndarray,
        y_test_raw: np.ndarray,
        centroids: np.ndarray,
        threshold: float,
        encoder: LabelEncoder,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用嵌入进行开集识别预测。

        Args:
            model: 训练好的 CNN 模型。
            x_test: 测试特征。
            y_test_raw: 测试集原始标签。
            centroids: 已知类别的质心。
            threshold: 距离阈值。
            encoder: 标签编码器。

        Returns:
            (y_pred_open, y_true_open) 元组。
        """
        test_embeddings = self._extract_embeddings(model, x_test)
        
        # 获取基础分类器的预测
        y_pred_enc = self._predict(model, x_test)
        
        # 计算到预测类质心的距离
        predicted_centroids = centroids[y_pred_enc]
        distances = np.linalg.norm(test_embeddings - predicted_centroids, axis=1)
        
        # 判断已知/未知
        is_known = distances <= threshold
        y_pred_open = np.where(is_known, "known", "unknown")
        
        # 确定真实标签
        known_classes_set = set(encoder.classes_)
        y_true_open = np.array(
            ["known" if label in known_classes_set else "unknown" for label in y_test_raw]
        )
        
        return y_pred_open, y_true_open

    def _save_checkpoint(self, checkpoint: CNNCheckpoint, filename: str) -> str:
        """保存模型检查点。

        Args:
            checkpoint: 检查点对象。

        Returns:
            保存的文件路径字符串。
        """
        path = self.pipeline_config.model_dir / filename
        torch.save(
            {
                "state_dict": checkpoint.state_dict,
                "config": checkpoint.config,
                "input_channels": checkpoint.input_channels,
                "num_classes": checkpoint.num_classes,
                "label_encoder": checkpoint.label_encoder,
                "centroids": checkpoint.centroids,
                "threshold": checkpoint.threshold,
                "task": checkpoint.task,
            },
            path,
        )
        return str(path)

    def _build_result(
        self,
        task: str,
        metrics: Dict[str, float],
        history: List[Dict],
        encoder: LabelEncoder,
        checkpoint_path: str,
        extra: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """构建训练结果字典。

        Args:
            task: 任务类型。
            metrics: 评估指标。
            history: 训练历史。
            encoder: 标签编码器。
            checkpoint_path: 检查点路径。
            extra: 额外信息。

        Returns:
            结果字典。
        """
        best_val_acc = max((h["val"]["accuracy"] for h in history), default=0.0)
        result: Dict[str, Any] = {
            task: {
                **metrics,
                "best_val_accuracy": best_val_acc,
            },
            "history": history,
            "checkpoint_path": checkpoint_path,
            "encoder_classes": encoder.classes_.tolist(),
        }
        if extra:
            result[task].update(extra)
        return result


# =============================================================================
# 推理与便捷函数
# =============================================================================


class CNNInference:
    """CNN 模型推理器。

    用于加载检查点并进行预测。

    Attributes:
        model: 加载的 CNN 模型。
        encoder: 标签编码器。
        device: 计算设备。
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """初始化推理器。

        Args:
            checkpoint_path: 检查点文件路径。
        """
        _require_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            # 兼容旧版 PyTorch 不支持 weights_only 参数的情况。
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]
        is_legacy_state = any(key.startswith("features.") for key in state_dict)
        legacy_config: Dict[str, Any] = {}
        modern_config: Optional[CNNConfig] = None

        if is_legacy_state:
            legacy_config = config if isinstance(config, dict) else {}
            self.config = legacy_config
        else:
            if isinstance(config, dict):
                modern_config = CNNConfig(
                    conv_channels=tuple(config.get("conv_channels", (32, 64, 128))),
                    kernel_size=int(config.get("kernel_size", 7)),
                    padding=int(config.get("padding", 3)),
                    hidden_units=int(config.get("hidden_units", 256)),
                    dropout_rates=tuple(config.get("dropout_rates", (0.2, 0.5))),
                    window_size=int(config.get("window_size", 200)),
                )
            else:
                modern_config = config
            self.config = modern_config
        self.input_channels = checkpoint["input_channels"]
        self.num_classes = checkpoint["num_classes"]

        encoder = checkpoint.get("label_encoder")
        if encoder is None:
            label_classes = checkpoint.get("label_classes") or checkpoint.get(
                "centroid_labels"
            )
            if label_classes is None:
                raise KeyError("检查点中缺少标签编码信息。")
            encoder = LabelEncoder()
            encoder.fit(list(label_classes))
        self.encoder = encoder

        self.centroids = checkpoint.get("centroids")
        if self.centroids is None:
            self.centroids = checkpoint.get("centroid_vectors")
        if self.centroids is not None:
            self.centroids = np.asarray(self.centroids, dtype=np.float32)
        threshold = checkpoint.get("threshold")
        self.threshold = float(threshold) if threshold is not None else None
        self.task = checkpoint.get("task", "classification")

        if is_legacy_state:
            self.model = LegacyRSSICNNClassifier(
                self.input_channels,
                self.num_classes,
                legacy_config,
                window_size=int(legacy_config.get("window_size", 200)),
            )
        else:
            if modern_config is None:
                raise ValueError("CNN 配置缺失，无法构建推理模型。")
            self.model = RSSICNNClassifier(
                self.input_channels, self.num_classes, modern_config
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """进行预测。

        Args:
            x: 输入特征 (N, W, C)。

        Returns:
            预测标签数组。
        """
        preds = self._predict_encoded(x)
        return self.encoder.inverse_transform(preds)

    def get_embeddings(self, x: np.ndarray) -> np.ndarray:
        """提取嵌入向量。

        Args:
            x: 输入特征 (N, W, C)。

        Returns:
            嵌入向量数组。
        """
        with torch.no_grad():
            tensor_x = (
                torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
            )
            embeddings = self.model.get_embeddings(tensor_x)
            return embeddings.cpu().numpy()

    def predict_open_set(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行开集推理，输出 known/unknown 标签。

        Args:
            x: 输入特征 (N, W, C)。

        Returns:
            (predictions, probabilities_or_scores) 元组。
        """
        if self.centroids is None or self.threshold is None:
            raise ValueError("当前检查点不包含开集元信息，无法执行开集推理。")

        embeddings = self.get_embeddings(x)
        y_pred_enc = self._predict_encoded(x)
        predicted_centroids = self.centroids[y_pred_enc]
        distances = np.linalg.norm(embeddings - predicted_centroids, axis=1)
        predictions = np.where(distances <= float(self.threshold), "known", "unknown")
        return predictions, distances

    def _predict_encoded(self, x: np.ndarray) -> np.ndarray:
        """预测编码标签，供开集推理内部复用。"""
        with torch.no_grad():
            tensor_x = (
                torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
            )
            logits = self.model(tensor_x)
            return torch.argmax(logits, dim=1).cpu().numpy()


def load_cnn_checkpoint(path: Path) -> CNNInference:
    """加载 CNN 检查点并返回推理器。

    Args:
        path: 检查点路径。

    Returns:
        CNNInference 实例。
    """
    return CNNInference(path)


def predict_cnn_windows(
    model_path: Path, windows: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """对滑动窗口数据进行预测。

    Args:
        model_path: 模型检查点路径。
        windows: 滑动窗口数据 (N, W, C)。

    Returns:
        (predictions, probabilities) 元组。
    """
    inference = load_cnn_checkpoint(model_path)
    preds = inference.predict(windows)
    
    # 获取概率
    with torch.no_grad():
        tensor_x = (
            torch.tensor(windows, dtype=torch.float32)
            .permute(0, 2, 1)
            .to(inference.device)
        )
        logits = inference.model(tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
    return preds, probs


def predict_cnn_probabilities(model_path: Path, windows: np.ndarray) -> np.ndarray:
    """预测概率。

    Args:
        model_path: 模型检查点路径。
        windows: 滑动窗口数据 (N, W, C)。

    Returns:
        概率数组。
    """
    _, probs = predict_cnn_windows(model_path, windows)
    return probs


def predict_cnn_embeddings(model_path: Path, windows: np.ndarray) -> np.ndarray:
    """预测嵌入向量。

    Args:
        model_path: 模型检查点路径。
        windows: 滑动窗口数据 (N, W, C)。

    Returns:
        嵌入向量数组。
    """
    inference = load_cnn_checkpoint(model_path)
    return inference.get_embeddings(windows)


def train_cnn_classification(
    data_file: Optional[Path] = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    early_stop_patience: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """训练 CNN 分类模型。

    Args:
        data_file: 数据文件路径。
        epochs: 训练轮数。
        batch_size: 批大小。
        learning_rate: 学习率。
        val_ratio: 验证集比例。
        early_stop_patience: 早停耐心轮数。
        seed: 随机种子。

    Returns:
        训练结果字典。
    """
    trainer = CNNTrainer(
        train_config=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            early_stop_patience=early_stop_patience,
            random_seed=seed,
        )
    )
    return trainer.train_classification(data_file)


def train_cnn_identification(
    data_file: Optional[Path] = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    early_stop_patience: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """训练 CNN 识别模型。

    Args:
        data_file: 数据文件路径。
        epochs: 训练轮数。
        batch_size: 批大小。
        learning_rate: 学习率。
        val_ratio: 验证集比例。
        early_stop_patience: 早停耐心轮数。
        seed: 随机种子。

    Returns:
        训练结果字典。
    """
    trainer = CNNTrainer(
        train_config=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            early_stop_patience=early_stop_patience,
            random_seed=seed,
        )
    )
    return trainer.train_identification(data_file)


# =============================================================================
# 命令行入口
# =============================================================================


def main() -> None:
    """主函数：解析命令行参数并执行 CNN 模型训练。"""
    parser = argparse.ArgumentParser(description="RSSI 1D CNN 模型训练工具")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "identification"],
        help="训练任务类型",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数，默认 20")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小，默认 64")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="学习率，默认 0.001"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="早停耐心轮数，默认 5",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    try:
        _require_torch()
    except ImportError as e:
        print(f"错误: {e}")
        return

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        random_seed=args.seed,
    )

    trainer = CNNTrainer(train_config=train_config)

    if args.task == "classification":
        result = trainer.train_classification()
        print("\nCNN 分类模型训练完成")
    else:
        result = trainer.train_identification()
        print("\nCNN 识别模型训练完成")
        
    _print_results(result, args.task)


def _print_results(result: dict, task: str) -> None:
    """打印训练结果。

    Args:
        result: 结果字典。
        task: 任务类型。
    """
    metrics = result.get(task, {})
    print(f"\n=== {task.upper()} 结果 ===")
    print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"  精确率: {metrics.get('precision', 0):.4f}")
    print(f"  召回率: {metrics.get('recall', 0):.4f}")
    print(f"  F1分数: {metrics.get('f1', 0):.4f}")
    print(f"  最佳验证准确率: {metrics.get('best_val_accuracy', 0):.4f}")
    if task == "identification":
        print(f"  距离阈值: {metrics.get('threshold', 0):.4f}")


if __name__ == "__main__":
    main()