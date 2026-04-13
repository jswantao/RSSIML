# -*- coding: utf-8 -*-
"""RSSI 传统模型训练与验证模块。

提供身份分类和开集识别的传统机器学习模型（如随机森林、SVM等）的训练、评估和保存功能。
 a.o.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, cast

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from scripts.config import PipelineConfig

# 定义公共 API
__all__ = [
    "ClassificationMetrics",
    "MetricsCalculator",
    "ModelFactory",
    "DataLoader",
    "ClassificationTrainer",
    "IdentificationModel",
    "IdentificationTrainer",
    "TrainingConfig",
    "TrainingPipeline",
    "train_classification",
    "train_identification",
]


# =============================================================================
# 数据类与协议
# =============================================================================


@dataclass(frozen=True)
class ClassificationMetrics:
    """分类任务评估指标。

    Attributes:
        accuracy: 准确率。
        precision: 精确率 (Macro Average)。
        recall: 召回率 (Macro Average)。
        f1: F1 分数 (Macro Average)。
        cv_accuracy: 交叉验证平均准确率（可选）。
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式。

        Returns:
            包含所有指标的字典。
        """
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.cv_accuracy is not None:
            result["cv_accuracy"] = self.cv_accuracy
        return result


@dataclass(frozen=True)
class TrainingConfig:
    """模型训练配置。

    Attributes:
        cv_folds: 交叉验证折数。
        validation_split: 验证集比例（保留字段，当前主要使用 CV）。
        distance_threshold_quantile: 开集识别距离阈值分位数。
        random_seed: 随机种子。
    """

    cv_folds: int = 5
    validation_split: float = 0.2
    distance_threshold_quantile: float = 0.95
    random_seed: int = 42

    def __post_init__(self) -> None:
        """验证配置参数的有效性。

        Raises:
            ValueError: 当参数超出合法范围时抛出。
        """
        if self.cv_folds < 2:
            raise ValueError(f"交叉验证折数必须大于等于2，实际为 {self.cv_folds}")
        if not 0 < self.validation_split < 1.0:
            raise ValueError(
                f"验证集比例必须在 (0, 1) 之间，实际为 {self.validation_split}"
            )
        if not 0 < self.distance_threshold_quantile <= 1.0:
            raise ValueError(
                f"距离阈值分位数必须在 (0, 1] 之间，实际为 {self.distance_threshold_quantile}"
            )


class SklearnClassifierProtocol(BaseEstimator, ClassifierMixin):
    """Sklearn 分类器协议，用于更严格的类型检查。

    继承自 BaseEstimator 和 ClassifierMixin，确保与 sklearn 的 cross_val_score 等函数兼容。
    同时要求实现 fit, predict 和 score 方法。
    """

    def fit(self, X: Any, y: Any) -> "SklearnClassifierProtocol":
        ...

    def predict(self, X: Any) -> Any:
        ...

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        ...


# =============================================================================
# 工具类
# =============================================================================


class MetricsCalculator:
    """评估指标计算器。"""

    # 定义合法的 average 策略类型
    _AverageStrategy = Literal["micro", "macro", "samples", "weighted", "binary"]

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: _AverageStrategy = "macro",
        pos_label: str = "1",
    ) -> ClassificationMetrics:
        """计算分类评估指标。

        Args:
            y_true: 真实标签。
            y_pred: 预测标签。
            average: 多分类平均策略，默认为 "macro"。
            pos_label: 当 average="binary" 时的正类标签。

        Returns:
            ClassificationMetrics 实例。
        """
        # 确保标签为字符串类型，以兼容 sklearn 的多分类指标计算
        y_true_str = y_true.astype(str)
        y_pred_str = y_pred.astype(str)

        metric_kwargs: Dict[str, Any] = {
            "average": average,
            "zero_division": 0,
        }
        if average == "binary":
            metric_kwargs["pos_label"] = str(pos_label)

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true_str, y_pred_str)),
            precision=float(precision_score(y_true_str, y_pred_str, **metric_kwargs)),
            recall=float(recall_score(y_true_str, y_pred_str, **metric_kwargs)),
            f1=float(f1_score(y_true_str, y_pred_str, **metric_kwargs)),
        )


class ModelFactory:
    """模型工厂，用于创建不同类型的分类器。

    Attributes:
        config: 训练配置，用于获取随机种子 a.o.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """初始化模型工厂。

        Args:
            config: 训练配置。
        """
        self.config = config

    def create_all(self) -> Dict[str, SklearnClassifierProtocol]:
        """创建所有预定义的分类器。

        Returns:
            模型名称到模型实例的映射。
        """
        seed = self.config.random_seed
        return {
            "RandomForest": cast(
                SklearnClassifierProtocol,
                RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            ),
            "LogisticRegression": cast(
                SklearnClassifierProtocol,
                LogisticRegression(random_state=seed, max_iter=1000, n_jobs=-1),
            ),
            "SVC": cast(
                SklearnClassifierProtocol,
                SVC(random_state=seed, probability=False),
            ),  # SVM 通常较慢，关闭概率估计加速
        }


class DataLoader:
    """数据加载器，负责加载预处理后的数据。

    Attributes:
        config: 流水线配置。
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """初始化数据加载器。

        Args:
            config: 流水线配置。若为 None，则自动推断。
        """
        self.config = config or PipelineConfig.from_root()

    def load_processed(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载预处理后的数据。

        Args:
            file_path: 数据文件路径，None 时使用默认路径。

        Returns:
            包含训练/测试特征和标签的字典。

        Raises:
            FileNotFoundError: 当数据文件不存在时抛出。
        """
        path = file_path or (self.config.data_dir / "rssi_processed.pkl")
        if not path.exists():
            raise FileNotFoundError(f"预处理数据文件不存在: {path}")

        with path.open("rb") as f:
            return pickle.load(f)


# =============================================================================
# 分类训练器
# =============================================================================


class ClassificationTrainer:
    """分类模型训练器。

    负责训练多个分类模型并评估其性能。

    Attributes:
        config: 流水线全局配置。
        train_config: 训练特定配置。
        factory: 模型工厂。
        loader: 数据加载器。
        label_encoder: 标签编码器。
    """

    def __init__(
        self,
        train_config: Optional[TrainingConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化分类训练器。

        Args:
            train_config: 训练配置。
            config: 流水线全局配置。
        """
        self.config = config or PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.factory = ModelFactory(self.train_config)
        self.loader = DataLoader(self.config)
        self.label_encoder = LabelEncoder()

    def train(self, data_file: Optional[Path] = None) -> Dict[str, ClassificationMetrics]:
        """训练并评估所有分类模型。

        Args:
            data_file: 预处理数据文件路径。

        Returns:
            模型名称到评估指标的映射字典。

        Raises:
            ValueError: 当训练集和测试集类别不匹配时抛出。
        """
        # 加载数据
        data = self.loader.load_processed(data_file)
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train_raw = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test_raw = np.asarray(data["y_test"], dtype=object)

        # 编码标签
        y_train_enc = np.asarray(
            self.label_encoder.fit_transform(y_train_raw), dtype=np.int64
        )
        try:
            y_test_enc = np.asarray(
                self.label_encoder.transform(y_test_raw), dtype=np.int64
            )
        except ValueError as e:
            raise ValueError(
                f"测试集包含训练集中未见过的标签: {e}. "
                f"请确保数据划分阶段已知/未知身份处理正确。"
            ) from e

        # 验证类别一致性
        self._validate_class_consistency(y_train_enc, y_test_enc)

        # 创建交叉验证器
        cv = StratifiedKFold(
            n_splits=self.train_config.cv_folds,
            shuffle=True,
            random_state=self.train_config.random_seed,
        )

        # 训练并评估所有模型
        results: Dict[str, ClassificationMetrics] = {}
        best_model_info: Optional[Tuple[str, SklearnClassifierProtocol, float]] = None

        for name, model in self.factory.create_all().items():
            metrics = self._train_and_evaluate(
                name, model, x_train, y_train_enc, x_test, y_test_enc, cv
            )
            results[name] = metrics

            # 更新最佳模型
            if best_model_info is None or metrics.accuracy > best_model_info[2]:
                best_model_info = (name, model, metrics.accuracy)

        # 保存最佳模型
        if best_model_info:
            self._save_model(
                best_model_info[1], f"best_{best_model_info[0]}_classification.joblib"
            )

        # 保存结果
        self._save_results(results)
        return results

    def _validate_class_consistency(
        self, y_train: np.ndarray, y_test: np.ndarray
    ) -> None:
        """验证训练集和测试集类别一致性。

        Args:
            y_train: 训练集编码标签。
            y_test: 测试集编码标签。

        Raises:
            ValueError: 当类别不一致时抛出。
        """
        train_classes = set(np.unique(y_train).tolist())
        test_classes = set(np.unique(y_test).tolist())

        if not test_classes.issubset(train_classes):
            missing = test_classes - train_classes
            raise ValueError(
                f"测试集包含训练集中未见的类别: {missing}. "
                f"这在闭集分类任务中是不允许的。"
            )

    def _train_and_evaluate(
        self,
        name: str,
        model: SklearnClassifierProtocol,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        cv: StratifiedKFold,
    ) -> ClassificationMetrics:
        """训练单个模型并评估。

        Args:
            name: 模型名称。
            model: 分类器实例。
            x_train: 训练特征。
            y_train: 训练标签。
            x_test: 测试特征。
            y_test: 测试标签。
            cv: 交叉验证器。

        Returns:
            评估指标。
        """
        # 交叉验证 (仅在训练集上进行，以评估泛化能力)
        # 注意：cross_val_score 期望 estimator 具有 fit/predict/score
        cv_scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )

        # 全量训练
        model.fit(x_train, y_train)

        # 测试集预测
        y_pred = model.predict(x_test)

        # 计算指标
        metrics = MetricsCalculator.calculate(y_test, y_pred, average="macro")

        #由于 ClassificationMetrics 是 frozen dataclass，我们需要通过替换整个对象来添加 cv_accuracy
        final_metrics = ClassificationMetrics(
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            cv_accuracy=float(np.mean(cv_scores)),
        )

        return final_metrics

    def _save_model(self, model: Any, filename: str) -> None:
        """保存模型。

        Args:
            model: 训练好的模型。
            filename: 文件名。
        """
        path = self.config.model_dir / filename
        joblib.dump(model, path)

    def _save_results(self, results: Dict[str, ClassificationMetrics]) -> None:
        """保存评估结果。

        Args:
            results: 模型名称到指标的映射。
        """
        output: Dict[str, Any] = {
            "models": {name: metrics.to_dict() for name, metrics in results.items()}
        }
        if results:
            best_name = max(results.keys(), key=lambda k: results[k].accuracy)
            output["best_model"] = best_name

        path = self.config.result_dir / "classification_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


# =============================================================================
# 开集识别训练器
# =============================================================================


@dataclass
class IdentificationModel:
    """开集识别模型。

    Attributes:
        classifier: 闭集分类器。
        centroids: 每个类别的质心向量。
        threshold: 开集识别距离阈值。
        label_encoder: 标签编码器。
    """

    classifier: SklearnClassifierProtocol
    centroids: np.ndarray
    threshold: float
    label_encoder: LabelEncoder


class IdentificationTrainer:
    """开集识别模型训练器。

    实现基于距离的开集识别，使用闭集分类器的决策边界或特征空间中心。

    Attributes:
        config: 流水线全局配置。
        train_config: 训练特定配置。
        factory: 模型工厂。
        loader: 数据加载器。
    """

    def __init__(
        self,
        train_config: Optional[TrainingConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化识别训练器。

        Args:
            train_config: 训练配置。
            config: 流水线全局配置。
        """
        self.config = config or PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.factory = ModelFactory(self.train_config)
        self.loader = DataLoader(self.config)
        self.label_encoder = LabelEncoder()

    def train(self, data_file: Optional[Path] = None) -> Dict[str, Any]:
        """训练开集识别模型。

        Args:
            data_file: 预处理数据文件路径。

        Returns:
            包含模型、阈值和评估结果的字典。
        """
        # 加载数据
        data = self.loader.load_processed(data_file)
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train_raw = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test_raw = np.asarray(data["y_test"], dtype=object)

        # 编码训练集标签
        y_train_enc = np.asarray(
            self.label_encoder.fit_transform(y_train_raw), dtype=np.int64
        )

        # 训练闭集分类器 (默认使用随机森林，因其对噪声鲁棒且能提供较好的特征空间分布)
        base_model = self.factory.create_all()["RandomForest"]
        base_model.fit(x_train, y_train_enc)

        # 计算质心
        unique_labels = np.unique(y_train_enc)
        centroids = self._compute_centroids(x_train, y_train_enc, unique_labels)

        # 计算距离阈值
        threshold = self._compute_threshold(
            x_train, y_train_enc, centroids, unique_labels
        )

        # 构建识别模型
        model = IdentificationModel(
            classifier=base_model,
            centroids=centroids,
            threshold=threshold,
            label_encoder=self.label_encoder,
        )

        # 评估模型
        y_pred_open, y_true_open = self._predict_identification(
            model, x_test, y_test_raw
        )

        # 计算指标 (二分类: Known vs Unknown)
        metrics = MetricsCalculator.calculate(
            y_true_open, y_pred_open, average="binary", pos_label="known"
        )

        # 保存模型和结果
        self._save_identification_model(model)
        self._save_identification_results(metrics)

        return {
            "model": model,
            "threshold": threshold,
            "centroids": centroids,
            "metrics": metrics.to_dict(),
        }

    @staticmethod
    def _compute_centroids(
        x: np.ndarray, y_enc: np.ndarray, unique_labels: np.ndarray
    ) -> np.ndarray:
        """计算每个类的质心。

        Args:
            x: 特征矩阵。
            y_enc: 编码后的标签。
            unique_labels: 唯一标签列表。

        Returns:
            质心矩阵，形状 (n_classes, n_features)。
        """
        n_classes = len(unique_labels)
        n_features = x.shape[1]
        centroids = np.zeros((n_classes, n_features), dtype=np.float32)

        for i, label in enumerate(unique_labels):
            mask = y_enc == label
            if np.any(mask):
                centroids[i] = np.mean(x[mask], axis=0)

        return centroids

    def _compute_threshold(
        self,
        x: np.ndarray,
        y_enc: np.ndarray,
        centroids: np.ndarray,
        unique_labels: np.ndarray,
    ) -> float:
        """计算开集识别的距离阈值。

        Args:
            x: 训练特征。
            y_enc: 训练标签。
            centroids: 类质心。
            unique_labels: 唯一标签。

        Returns:
            距离阈值。
        """
        distances = []
        label_to_idx = {int(label): idx for idx, label in enumerate(unique_labels)}

        for i in range(len(x)):
            label = int(y_enc[i])
            centroid = centroids[label_to_idx[label]]
            dist = np.linalg.norm(x[i] - centroid)
            distances.append(dist)

        if not distances:
            return 0.0

        return float(
            np.quantile(distances, self.train_config.distance_threshold_quantile)
        )

    def _predict_identification(
        self, model: IdentificationModel, x: np.ndarray, y_raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行开集识别预测。

        Args:
            model: 训练好的识别模型。
            x: 测试特征。
            y_raw: 测试集原始标签。

        Returns:
            (预测标签, 最终真实标签) 元组。标签为 "known" 或 "unknown"。
        """
        # 获取闭集预测 the classifier returns encoded labels
        y_pred_enc = model.classifier.predict(x)

        # 计算到预测类质心的距离
        centroids = model.centroids
        # 向量化计算距离
        predicted_centroids = centroids[y_pred_enc]
        distances = np.linalg.norm(x - predicted_centroids, axis=1)

        # 判断已知/未知
        is_known = distances <= model.threshold
        y_pred_open = np.where(is_known, "known", "unknown")

        # 确定真实标签
        # 如果原始标签在训练集中出现过，则为 known，否则为 unknown
        known_classes_set = set(model.label_encoder.classes_)
        y_true_open = np.array(
            ["known" if label in known_classes_set else "unknown" for label in y_raw]
        )

        return y_pred_open, y_true_open

    def _save_identification_model(self, model: IdentificationModel) -> None:
        """保存开集识别模型。

        Args:
            model: 识别模型实例。
        """
        path = self.config.model_dir / "identification_model.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_identification_results(self, metrics: ClassificationMetrics) -> None:
        """保存识别任务评估结果。

        Args:
            metrics: 评估指标。
        """
        path = self.config.result_dir / "identification_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics.to_dict()}, f, indent=2, ensure_ascii=False)


# =============================================================================
# 流水线管理
# =============================================================================


class TrainingPipeline:
    """模型训练流水线。

    整合分类和识别任务的训练流程。

    Attributes:
        config: 流水线配置。
        train_config: 训练配置。
        classification_trainer: 分类训练器。
        identification_trainer: 识别训练器。
    """

    def __init__(
        self,
        train_config: Optional[TrainingConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化训练流水线。

        Args:
            train_config: 训练配置。
            config: 流水线配置。
        """
        self.config = config or PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.classification_trainer = ClassificationTrainer(
            self.train_config, self.config
        )
        self.identification_trainer = IdentificationTrainer(
            self.train_config, self.config
        )

    def run_classification(self, data_file: Optional[Path] = None) -> dict:
        """运行分类模型训练。

        Args:
            data_file: 数据文件路径。

        Returns:
            分类结果字典。
        """
        results = self.classification_trainer.train(data_file)
        return {name: metrics.to_dict() for name, metrics in results.items()}

    def run_identification(self, data_file: Optional[Path] = None) -> dict:
        """运行识别模型训练。

        Args:
            data_file: 数据文件路径。

        Returns:
            识别结果字典。
        """
        return self.identification_trainer.train(data_file)

    def run_all(
        self,
        classification_file: Optional[Path] = None,
        identification_file: Optional[Path] = None,
    ) -> dict:
        """运行全部训练任务。

        Args:
            classification_file: 分类任务数据文件。
            identification_file: 识别任务数据文件。

        Returns:
            包含所有结果的字典。
        """
        return {
            "classification": self.run_classification(classification_file),
            "identification": self.run_identification(identification_file),
        }


# =============================================================================
# 便捷函数
# =============================================================================


def train_classification(
    data_file: Optional[Path] = None,
    train_config: Optional[TrainingConfig] = None,
) -> Dict[str, float]:
    """训练分类模型并返回指标。

    Args:
        data_file: 预处理数据文件。
        train_config: 训练配置。

    Returns:
        模型名称到指标的映射字典。
    """
    pipeline = TrainingPipeline(train_config)
    return pipeline.run_classification(data_file)


def train_identification(
    data_file: Optional[Path] = None,
    train_config: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    """训练识别模型并返回结果。

    Args:
        data_file: 预处理数据文件。
        train_config: 训练配置。

    Returns:
        识别结果字典。
    """
    pipeline = TrainingPipeline(train_config)
    return pipeline.run_identification(data_file)


# =============================================================================
# 命令行入口
# =============================================================================


def main() -> None:
    """主函数：解析命令行参数并执行模型训练。"""
    parser = argparse.ArgumentParser(description="RSSI 传统模型训练工具")
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["classification", "identification", "both"],
        help="训练任务类型",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="交叉验证折数，默认 5"
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="开集识别距离阈值分位数，默认 0.95",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    # 创建训练配置
    train_config = TrainingConfig(
        cv_folds=args.cv_folds,
        distance_threshold_quantile=args.threshold_quantile,
        random_seed=args.seed,
    )
    pipeline = TrainingPipeline(train_config)

    if args.task == "classification":
        result = pipeline.run_classification()
        print("\n分类模型训练完成")
        _print_classification_results(result)
    elif args.task == "identification":
        result = pipeline.run_identification()
        print("\n识别模型训练完成")
        _print_identification_results(result)
    else:
        result = pipeline.run_all()
        print("\n全部模型训练完成")
        print("\n=== 分类模型 ===")
        _print_classification_results(result["classification"])
        print("\n=== 识别模型 ===")
        _print_identification_results(result["identification"])


def _print_classification_results(results: Dict[str, Dict[str, float]]) -> None:
    """打印分类结果。

    Args:
        results: 分类结果字典。
    """
    for name, metrics in results.items():
        print(
            f"{name}: Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )


def _print_identification_results(result: Dict[str, Any]) -> None:
    """打印识别结果。

    Args:
        result: 识别结果字典。
    """
    metrics = result.get("metrics", {})
    print(f"Accuracy (Known vs Unknown): {metrics.get('accuracy', 0):.4f}")
    print(f"F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"Threshold: {result.get('threshold', 0):.4f}")


if __name__ == "__main__":
    main()