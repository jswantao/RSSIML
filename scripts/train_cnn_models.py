"""RSSI 1D CNN 模型训练模块。

提供基于卷积神经网络的 RSSI 时序数据分类和开集识别功能。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from scripts.config import PipelineConfig


# ============================================================================
# 类型定义与配置
# ============================================================================


@dataclass(frozen=True)
class CNNConfig:
    """CNN 模型配置。
    
    Attributes:
        conv_channels: 卷积层通道数列表。
        kernel_sizes: 卷积核大小列表。
        dropout_rates: Dropout 比率列表。
        fc_units: 全连接层单元数。
    """
    conv_channels: Tuple[int, ...] = (64, 128, 256)
    kernel_sizes: Tuple[int, ...] = (5, 3, 3)
    dropout_rates: Tuple[float, ...] = (0.3, 0.2)
    fc_units: int = 128


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
        """验证配置参数的有效性。"""
        if not 0 < self.val_ratio < 0.5:
            raise ValueError(f"验证集比例必须在 (0, 0.5) 之间，实际为 {self.val_ratio}")
        if self.epochs < 1:
            raise ValueError(f"训练轮数必须大于 0，实际为 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"批大小必须大于 0，实际为 {self.batch_size}")


@dataclass(frozen=True)
class TrainingMetrics:
    """训练指标。
    
    Attributes:
        accuracy: 准确率。
        precision: 精确率。
        recall: 召回率。
        f1: F1 分数。
        loss: 损失值（可选）。
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式。"""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.loss is not None:
            result["loss"] = self.loss
        return result


@dataclass
class CNNCheckpoint:
    """CNN 模型检查点。
    
    Attributes:
        state_dict: 模型状态字典。
        config: CNN 配置。
        input_channels: 输入通道数。
        num_classes: 类别数。
        label_classes: 标签类别列表。
        task: 任务类型。
    """
    state_dict: Dict[str, torch.Tensor]
    config: CNNConfig
    input_channels: int
    num_classes: int
    label_classes: List[str]
    task: str = "classification"
    
    # 开集识别特有字段
    threshold: Optional[float] = None
    centroid_labels: Optional[List[str]] = None
    centroid_vectors: Optional[np.ndarray] = None
    
    def save(self, path: Path) -> None:
        """保存检查点到文件。
        
        Args:
            path: 保存路径。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "state_dict": self.state_dict,
            "config": {
                "conv_channels": list(self.config.conv_channels),
                "kernel_sizes": list(self.config.kernel_sizes),
                "dropout_rates": list(self.config.dropout_rates),
                "fc_units": self.config.fc_units,
            },
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "label_classes": self.label_classes,
            "task": self.task,
        }
        
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.centroid_labels is not None:
            payload["centroid_labels"] = self.centroid_labels
        if self.centroid_vectors is not None:
            payload["centroid_vectors"] = self.centroid_vectors.tolist()
        
        torch.save(payload, path)
    
    @classmethod
    def load(
        cls,
        path: Path,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> CNNCheckpoint:
        """从文件加载检查点。
        
        Args:
            path: 检查点文件路径。
            map_location: 设备映射。
            
        Returns:
            CNNCheckpoint 实例。
        """
        payload = torch.load(path, map_location=map_location)
        
        config = CNNConfig(
            conv_channels=tuple(payload["config"]["conv_channels"]),
            kernel_sizes=tuple(payload["config"]["kernel_sizes"]),
            dropout_rates=tuple(payload["config"]["dropout_rates"]),
            fc_units=payload["config"]["fc_units"],
        )
        
        checkpoint = cls(
            state_dict=payload["state_dict"],
            config=config,
            input_channels=payload["input_channels"],
            num_classes=payload["num_classes"],
            label_classes=payload["label_classes"],
            task=payload.get("task", "classification"),
        )
        
        if "threshold" in payload:
            checkpoint.threshold = payload["threshold"]
        if "centroid_labels" in payload:
            checkpoint.centroid_labels = payload["centroid_labels"]
        if "centroid_vectors" in payload:
            checkpoint.centroid_vectors = np.asarray(payload["centroid_vectors"], dtype=np.float32)
        
        return checkpoint


# ============================================================================
# 数据集与数据加载
# ============================================================================


class RSSIWindowDataset(Dataset):
    """RSSI 滑动窗口数据集。
    
    将 NumPy 数组包装为 PyTorch Dataset，支持按通道优先的格式转换。
    
    Attributes:
        windows: 窗口数据，形状 (n_samples, window_size, n_channels)。
        labels: 标签数组。
    """
    
    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        """初始化数据集。
        
        Args:
            windows: 三维窗口数据。
            labels: 标签数组。
            
        Raises:
            ValueError: 当输入维度不为 3 时抛出。
        """
        if windows.ndim != 3:
            raise ValueError(f"窗口数据维度应为 3，实际为 {windows.shape}")
        
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = np.asarray(labels)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本。
        
        Args:
            idx: 样本索引。
            
        Returns:
            (features, label) 元组，features 形状为 (n_channels, window_size)。
        """
        # 转换为 (n_channels, window_size) 格式
        features = torch.from_numpy(self.windows[idx].T.copy())
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return features, label


class DataLoaderFactory:
    """数据加载器工厂。
    
    Attributes:
        config: 训练配置。
    """
    
    def __init__(self, config: TrainingConfig) -> None:
        """初始化工厂。
        
        Args:
            config: 训练配置。
        """
        self.config = config
    
    def create(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器。
        
        Args:
            x_train: 训练特征。
            y_train: 训练标签。
            x_val: 验证特征。
            y_val: 验证标签。
            
        Returns:
            (train_loader, val_loader) 元组。
        """
        train_dataset = RSSIWindowDataset(x_train, y_train)
        val_dataset = RSSIWindowDataset(x_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        return train_loader, val_loader


# ============================================================================
# CNN 模型
# ============================================================================


class RSSICNNClassifier(nn.Module):
    """RSSI 时序 1D CNN 分类器。
    
    使用多个卷积层提取时序特征，通过全连接层进行分类。
    
    Attributes:
        config: CNN 配置。
        input_channels: 输入通道数。
        num_classes: 类别数。
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        config: Optional[CNNConfig] = None,
    ) -> None:
        """初始化 CNN 模型。
        
        Args:
            input_channels: 输入通道数（特征维度）。
            num_classes: 输出类别数。
            config: CNN 配置，None 时使用默认配置。
        """
        super().__init__()
        self.config = config or CNNConfig()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.features = self._build_conv_layers()
        self.classifier = self._build_fc_layers()
    
    def _build_conv_layers(self) -> nn.Sequential:
        """构建卷积特征提取层。
        
        Returns:
            卷积层序列。
        """
        layers = []
        in_channels = self.input_channels
        
        for i, (out_channels, kernel_size) in enumerate(
            zip(self.config.conv_channels, self.config.kernel_sizes)
        ):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ])
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        return nn.Sequential(*layers)
    
    def _build_fc_layers(self) -> nn.Sequential:
        """构建全连接分类层。
        
        Returns:
            全连接层序列。
        """
        in_features = self.config.conv_channels[-1]
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rates[0]),
            nn.Linear(in_features, self.config.fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rates[1]),
            nn.Linear(self.config.fc_units, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状 (batch_size, n_channels, window_size)。
            
        Returns:
            分类 logits，形状 (batch_size, num_classes)。
        """
        x = self.features(x)
        return self.classifier(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取分类层之前的特征向量。
        
        Args:
            x: 输入张量。
            
        Returns:
            特征向量，形状 (batch_size, feature_dim)。
        """
        x = self.features(x)
        return torch.flatten(x, start_dim=1)


# ============================================================================
# 训练器
# ============================================================================


class MetricsCalculator:
    """评估指标计算器。"""
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
    ) -> TrainingMetrics:
        """计算分类评估指标。
        
        Args:
            y_true: 真实标签。
            y_pred: 预测标签。
            average: 平均方式。
            
        Returns:
            TrainingMetrics 对象。
        """
        return TrainingMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        )


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
    ) -> None:
        """初始化训练器。
        
        Args:
            model_config: CNN 模型配置。
            train_config: 训练配置。
        """
        self.pipeline_config = PipelineConfig.from_root()
        self.model_config = model_config or CNNConfig()
        self.train_config = train_config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(self.train_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.train_config.random_seed)
    
    def _load_windowed_data(self, file_path: Optional[Path] = None) -> dict:
        """加载滑窗数据。
        
        Args:
            file_path: 数据文件路径。
            
        Returns:
            数据字典。
        """
        import pickle
        path = file_path or self.pipeline_config.data_dir / "rssi_windowed.pkl"
        with path.open("rb") as f:
            return pickle.load(f)
    
    def _prepare_data(
        self,
        data: dict,
        task: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """准备训练和测试数据。
        
        Args:
            data: 原始数据字典。
            task: 任务类型。
            
        Returns:
            (x_train, y_train_enc, x_test, y_test_enc, encoder) 元组。
        """
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=object)
        
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        
        if task == "classification":
            y_test_enc = encoder.transform(y_test)
            self._validate_class_consistency(y_train_enc, y_test_enc)
        else:
            known_subjects = set(y_train.tolist())
            y_test_enc = np.asarray([
                encoder.transform([label])[0] if label in known_subjects else -1
                for label in y_test.tolist()
            ])
        
        return x_train, y_train_enc, x_test, y_test_enc, encoder
    
    def _validate_class_consistency(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """验证训练集和测试集类别一致性。
        
        Args:
            y_train: 训练集标签。
            y_test: 测试集标签。
            
        Raises:
            ValueError: 当类别不一致时抛出。
        """
        train_classes = set(y_train.tolist())
        test_classes = set(y_test.tolist())
        
        if train_classes != test_classes:
            raise ValueError(
                f"训练集和测试集类别不一致: "
                f"训练集 {sorted(train_classes)}, 测试集 {sorted(test_classes)}"
            )
    
    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
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
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
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
    ) -> Dict[str, object]:
        """训练 CNN 分类模型。
        
        Args:
            data_file: 数据文件路径。
            
        Returns:
            训练结果字典。
        """
        # 加载和准备数据
        data = self._load_windowed_data(data_file)
        x_train, y_train, x_test, y_test, encoder = self._prepare_data(data, "classification")
        
        # 划分训练验证集
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train,
            test_size=self.train_config.val_ratio,
            random_state=self.train_config.random_seed,
            stratify=y_train,
        )
        
        # 创建数据加载器
        factory = DataLoaderFactory(self.train_config)
        train_loader, val_loader = factory.create(x_fit, y_fit, x_val, y_val)
        
        # 构建模型
        input_channels = x_train.shape[2]
        num_classes = len(encoder.classes_)
        model = RSSICNNClassifier(
            input_channels, num_classes, self.model_config
        ).to(self.device)
        
        # 训练
        history = self._train_model(model, train_loader, val_loader)
        
        # 测试评估
        test_preds = self._predict(model, x_test)
        test_metrics = MetricsCalculator.calculate(y_test, test_preds)
        
        # 保存检查点
        checkpoint = CNNCheckpoint(
            state_dict=model.state_dict(),
            config=self.model_config,
            input_channels=input_channels,
            num_classes=num_classes,
            label_classes=[str(c) for c in encoder.classes_],
            task="classification",
        )
        checkpoint.save(self.pipeline_config.model_dir / "cnn_classification.pt")
        
        return self._build_result(
            task="classification",
            metrics=test_metrics,
            history=history,
            encoder=encoder,
            checkpoint_path=str(self.pipeline_config.model_dir / "cnn_classification.pt"),
        )
    
    def train_identification(
        self,
        data_file: Optional[Path] = None,
    ) -> Dict[str, object]:
        """训练 CNN 开集识别模型。
        
        Args:
            data_file: 数据文件路径。
            
        Returns:
            训练结果字典。
        """
        # 加载和准备数据
        data = self._load_windowed_data(data_file)
        x_train, y_train, x_test, y_test, encoder = self._prepare_data(data, "identification")
        
        # 划分训练验证集
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train,
            test_size=self.train_config.val_ratio,
            random_state=self.train_config.random_seed,
            stratify=y_train,
        )
        
        # 创建数据加载器
        factory = DataLoaderFactory(self.train_config)
        train_loader, val_loader = factory.create(x_fit, y_fit, x_val, y_val)
        
        # 构建模型
        input_channels = x_train.shape[2]
        num_classes = len(encoder.classes_)
        model = RSSICNNClassifier(
            input_channels, num_classes, self.model_config
        ).to(self.device)
        
        # 训练
        history = self._train_model(model, train_loader, val_loader)
        
        # 提取嵌入特征
        fit_emb = self._extract_embeddings(model, x_fit)
        val_emb = self._extract_embeddings(model, x_val)
        test_emb = self._extract_embeddings(model, x_test)
        
        # 计算类中心和阈值
        centroids, centroid_labels = self._compute_centroids(fit_emb, y_fit, encoder)
        threshold = self._estimate_threshold(val_emb, y_val, centroids)
        
        # 开集预测
        test_preds, test_distances = self._predict_open_set(
            test_emb, centroids, centroid_labels, threshold
        )
        
        # 计算指标
        y_test_open = self._to_open_set_labels(y_test, encoder, y_train)
        test_metrics = MetricsCalculator.calculate(y_test_open, test_preds)
        
        # 保存检查点
        checkpoint = CNNCheckpoint(
            state_dict=model.state_dict(),
            config=self.model_config,
            input_channels=input_channels,
            num_classes=num_classes,
            label_classes=[str(c) for c in encoder.classes_],
            task="identification",
            threshold=threshold,
            centroid_labels=centroid_labels,
            centroid_vectors=centroids,
        )
        checkpoint.save(self.pipeline_config.model_dir / "cnn_identification.pt")
        
        return self._build_result(
            task="identification",
            metrics=test_metrics,
            history=history,
            encoder=encoder,
            checkpoint_path=str(self.pipeline_config.model_dir / "cnn_identification.pt"),
            extra={
                "threshold": threshold,
                "centroid_labels": centroid_labels,
                "mean_min_distance": float(np.mean(test_distances)),
            },
        )
    
    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict[str, float]]:
        """训练模型。
        
        Args:
            model: 模型。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            
        Returns:
            训练历史记录列表。
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
            train_metrics = MetricsCalculator.calculate(train_targets, train_preds)
            train_metrics.loss = train_loss
            
            # 验证
            val_loss, val_targets, val_preds = self._run_epoch(
                model, val_loader, criterion, None
            )
            val_metrics = MetricsCalculator.calculate(val_targets, val_preds)
            val_metrics.loss = val_loss
            
            scheduler.step(val_metrics.accuracy)
            
            history.append({
                "epoch": epoch + 1,
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
            })
            
            if val_metrics.accuracy > best_acc:
                best_acc = val_metrics.accuracy
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.train_config.early_stop_patience:
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return history
    
    def _predict(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """批量预测。
        
        Args:
            model: 模型。
            x: 输入数据。
            
        Returns:
            预测标签数组。
        """
        tensor_x = torch.from_numpy(x.transpose(0, 2, 1).copy()).to(self.device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds
    
    def _extract_embeddings(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """提取嵌入特征。
        
        Args:
            model: 模型。
            x: 输入数据。
            
        Returns:
            嵌入特征数组。
        """
        tensor_x = torch.from_numpy(x.transpose(0, 2, 1).copy()).to(self.device)
        model.eval()
        with torch.no_grad():
            embeddings = model.extract_features(tensor_x).cpu().numpy()
        return embeddings.astype(np.float32)
    
    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        encoder: LabelEncoder,
    ) -> Tuple[np.ndarray, List[str]]:
        """计算各类别的嵌入中心。
        
        Args:
            embeddings: 嵌入特征。
            labels: 标签。
            encoder: 标签编码器。
            
        Returns:
            (centroids, centroid_labels) 元组。
        """
        centroid_labels = [str(c) for c in encoder.classes_]
        centroids = []
        
        for i in range(len(encoder.classes_)):
            mask = labels == i
            centroids.append(np.mean(embeddings[mask], axis=0))
        
        return np.asarray(centroids, dtype=np.float32), centroid_labels
    
    def _estimate_threshold(
        self,
        val_emb: np.ndarray,
        val_labels: np.ndarray,
        centroids: np.ndarray,
        quantile: float = 0.95,
    ) -> float:
        """估计开集识别距离阈值。
        
        Args:
            val_emb: 验证集嵌入。
            val_labels: 验证集标签。
            centroids: 类中心。
            quantile: 分位数。
            
        Returns:
            距离阈值。
        """
        distances = np.linalg.norm(val_emb - centroids[val_labels], axis=1)
        return float(np.quantile(distances, quantile))
    
    def _predict_open_set(
        self,
        test_emb: np.ndarray,
        centroids: np.ndarray,
        centroid_labels: List[str],
        threshold: float,
        unknown_label: str = "Unknown",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """开集识别预测。
        
        Args:
            test_emb: 测试集嵌入。
            centroids: 类中心。
            centroid_labels: 类标签。
            threshold: 距离阈值。
            unknown_label: 未知标签。
            
        Returns:
            (predictions, distances) 元组。
        """
        distances = np.linalg.norm(
            test_emb[:, None, :] - centroids[None, :, :], axis=2
        )
        min_dist = np.min(distances, axis=1)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_label = np.asarray(centroid_labels)[nearest_idx]
        
        predictions = np.asarray([
            label if dist <= threshold else unknown_label
            for label, dist in zip(nearest_label, min_dist)
        ], dtype=object)
        
        return predictions, min_dist
    
    def _to_open_set_labels(
        self,
        y_test: np.ndarray,
        encoder: LabelEncoder,
        y_train: np.ndarray,
        unknown_label: str = "Unknown",
    ) -> np.ndarray:
        """将测试标签转换为开集标签。
        
        Args:
            y_test: 测试标签（原始）。
            encoder: 标签编码器。
            y_train: 训练标签。
            unknown_label: 未知标签。
            
        Returns:
            开集标签数组。
        """
        known_subjects = set(y_train.tolist())
        return np.asarray([
            label if label in known_subjects else unknown_label
            for label in y_test.tolist()
        ], dtype=object)
    
    def _build_result(
        self,
        task: str,
        metrics: TrainingMetrics,
        history: List[Dict],
        encoder: LabelEncoder,
        checkpoint_path: str,
        extra: Optional[Dict] = None,
    ) -> Dict[str, object]:
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
        result = {
            task: {
                **metrics.to_dict(),
                "best_val_accuracy": max(
                    h["val"]["accuracy"] for h in history
                ) if history else 0.0,
                "label_classes": [str(c) for c in encoder.classes_],
                "checkpoint": checkpoint_path,
                "history": history,
            }
        }
        
        if extra:
            result[task].update(extra)
        
        # 保存结果
        metrics_path = self.pipeline_config.result_dir / f"cnn_{task}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result


# ============================================================================
# 推理接口
# ============================================================================


class CNNInference:
    """CNN 模型推理接口。
    
    Attributes:
        checkpoint: 模型检查点。
        model: 加载的模型。
        device: 计算设备。
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """初始化推理接口。
        
        Args:
            checkpoint_path: 检查点文件路径。
            device: 计算设备。
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = CNNCheckpoint.load(checkpoint_path, map_location=self.device)
        self.model = self._build_model()
        self.model.eval()
    
    def _build_model(self) -> RSSICNNClassifier:
        """从检查点构建模型。
        
        Returns:
            加载权重的模型。
        """
        model = RSSICNNClassifier(
            input_channels=self.checkpoint.input_channels,
            num_classes=self.checkpoint.num_classes,
            config=self.checkpoint.config,
        ).to(self.device)
        model.load_state_dict(self.checkpoint.state_dict)
        return model
    
    def predict(self, windows: np.ndarray) -> np.ndarray:
        """预测窗口数据的类别。
        
        Args:
            windows: 输入窗口，形状 (n_samples, window_size, n_channels)。
            
        Returns:
            预测标签数组。
            
        Raises:
            ValueError: 当输入维度不为 3 时抛出。
        """
        if windows.ndim != 3:
            raise ValueError(f"输入维度应为 3，实际为 {windows.shape}")
        
        tensor_x = torch.from_numpy(
            windows.transpose(0, 2, 1).copy()
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_x)
            pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
        
        return np.asarray([self.checkpoint.label_classes[i] for i in pred_indices])
    
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """预测窗口数据的类别概率。
        
        Args:
            windows: 输入窗口。
            
        Returns:
            概率数组，形状 (n_samples, num_classes)。
        """
        if windows.ndim != 3:
            raise ValueError(f"输入维度应为 3，实际为 {windows.shape}")
        
        tensor_x = torch.from_numpy(
            windows.transpose(0, 2, 1).copy()
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probs
    
    def extract_embeddings(self, windows: np.ndarray) -> np.ndarray:
        """提取窗口数据的嵌入特征。
        
        Args:
            windows: 输入窗口。
            
        Returns:
            嵌入特征数组。
        """
        if windows.ndim != 3:
            raise ValueError(f"输入维度应为 3，实际为 {windows.shape}")
        
        tensor_x = torch.from_numpy(
            windows.transpose(0, 2, 1).copy()
        ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.extract_features(tensor_x).cpu().numpy()
        
        return embeddings.astype(np.float32)
    
    def predict_open_set(self, windows: np.ndarray) -> np.ndarray:
        """开集识别预测。
        
        Args:
            windows: 输入窗口。
            
        Returns:
            预测标签数组（包含 Unknown）。
            
        Raises:
            ValueError: 当模型不是识别任务时抛出。
        """
        if self.checkpoint.task != "identification":
            raise ValueError("当前模型不是开集识别模型")
        if self.checkpoint.threshold is None or self.checkpoint.centroid_vectors is None:
            raise ValueError("模型缺少开集识别所需参数")
        
        embeddings = self.extract_embeddings(windows)
        centroids = self.checkpoint.centroid_vectors
        centroid_labels = self.checkpoint.centroid_labels or []
        
        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :], axis=2
        )
        min_dist = np.min(distances, axis=1)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_label = np.asarray(centroid_labels)[nearest_idx]
        
        return np.asarray([
            label if dist <= self.checkpoint.threshold else "Unknown"
            for label, dist in zip(nearest_label, min_dist)
        ], dtype=object)


# ============================================================================
# 命令行入口
# ============================================================================


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数，默认 20",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小，默认 64",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学习率，默认 0.001",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例，默认 0.2",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="早停耐心轮数，默认 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    args = parser.parse_args()
    
    # 创建配置
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        random_seed=args.seed,
    )
    
    # 训练
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
        print(f"  拒识阈值: {metrics.get('threshold', 0):.4f}")
        print(f"  平均最小距离: {metrics.get('mean_min_distance', 0):.4f}")
    
    print(f"  检查点: {metrics.get('checkpoint', 'N/A')}")


if __name__ == "__main__":
    main()