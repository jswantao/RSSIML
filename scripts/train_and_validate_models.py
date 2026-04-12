"""RSSI 模型训练模块。

提供身份分类和开集识别的模型训练、评估和保存功能。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scripts.config import PipelineConfig


# ============================================================================
# 类型定义
# ============================================================================


class Classifier(Protocol):
    """分类器协议，定义统一的模型接口。"""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ClassificationMetrics:
    """分类评估指标。
    
    Attributes:
        accuracy: 准确率。
        precision: 精确率。
        recall: 召回率。
        f1: F1 分数。
        cv_accuracy: 交叉验证准确率（可选）。
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式。"""
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
        validation_split: 验证集比例。
        distance_threshold_quantile: 距离阈值分位数。
        random_seed: 随机种子。
    """
    cv_folds: int = 5
    validation_split: float = 0.2
    distance_threshold_quantile: float = 0.95
    random_seed: int = 42


# ============================================================================
# 评估指标
# ============================================================================


class MetricsCalculator:
    """评估指标计算器。"""
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
    ) -> ClassificationMetrics:
        """计算分类评估指标。
        
        Args:
            y_true: 真实标签数组。
            y_pred: 预测标签数组。
            average: 平均方式，"macro" 用于多分类，"binary" 用于二分类。
            
        Returns:
            ClassificationMetrics 对象。
        """
        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        )


# ============================================================================
# 数据加载器
# ============================================================================


class DataLoader:
    """数据加载器。
    
    Attributes:
        config: 流水线配置对象。
    """
    
    def __init__(self) -> None:
        """初始化数据加载器。"""
        self.config = PipelineConfig.from_root()
    
    def load_processed(self, file_path: Optional[Path] = None) -> dict:
        """加载预处理后的数据。
        
        Args:
            file_path: 数据文件路径，None 时使用默认路径。
            
        Returns:
            包含预处理数据的字典。
        """
        path = file_path or self.config.data_dir / "rssi_processed.pkl"
        with path.open("rb") as f:
            return joblib.load(f)  # type: ignore
    
    def load_split(self, file_path: Optional[Path] = None) -> dict:
        """加载数据划分文件。
        
        Args:
            file_path: 划分文件路径，None 时使用默认路径。
            
        Returns:
            包含划分数据的字典。
        """
        path = file_path or self.config.data_dir / "rssi_split.pkl"
        with path.open("rb") as f:
            return joblib.load(f)  # type: ignore


# ============================================================================
# 模型工厂
# ============================================================================


class ModelFactory:
    """分类模型工厂。
    
    Attributes:
        config: 训练配置。
        random_seed: 随机种子。
    """
    
    def __init__(self, config: TrainingConfig) -> None:
        """初始化模型工厂。
        
        Args:
            config: 训练配置。
        """
        self.config = config
        self.random_seed = config.random_seed
    
    def create_all(self) -> Dict[str, Classifier]:
        """创建所有分类模型。
        
        Returns:
            模型名称到模型实例的映射字典。
        """
        return {
            "svm": self._create_svm(),
            "random_forest": self._create_random_forest(),
            "xgboost": self._create_xgboost(),
            "lightgbm": self._create_lightgbm(),
        }
    
    def _create_svm(self) -> SVC:
        """创建 SVM 分类器。"""
        return SVC(
            kernel="rbf",
            C=6.0,
            gamma="scale",
            probability=True,
            random_state=self.random_seed,
        )
    
    def _create_random_forest(self) -> RandomForestClassifier:
        """创建随机森林分类器。"""
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=self.random_seed,
            n_jobs=-1,
        )
    
    def _create_xgboost(self) -> XGBClassifier:
        """创建 XGBoost 分类器。"""
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.random_seed,
            n_jobs=-1,
        )
    
    def _create_lightgbm(self) -> LGBMClassifier:
        """创建 LightGBM 分类器。"""
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=-1,
        )


# ============================================================================
# 分类训练器
# ============================================================================


class ClassificationTrainer:
    """身份分类模型训练器。
    
    Attributes:
        config: 流水线配置对象。
        train_config: 训练配置。
        loader: 数据加载器。
        factory: 模型工厂。
        label_encoder: 标签编码器。
    """
    
    def __init__(self, train_config: Optional[TrainingConfig] = None) -> None:
        """初始化分类训练器。
        
        Args:
            train_config: 训练配置，None 时使用默认配置。
        """
        self.config = PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.loader = DataLoader()
        self.factory = ModelFactory(self.train_config)
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
        y_train = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=object)
        
        # 编码标签
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        
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
        best_model: Optional[Tuple[str, Classifier, float]] = None
        
        for name, model in self.factory.create_all().items():
            metrics = self._train_and_evaluate(
                name, model, x_train, y_train_enc, x_test, y_test_enc, cv
            )
            results[name] = metrics
            
            # 更新最佳模型
            if best_model is None or metrics.accuracy > best_model[2]:
                best_model = (name, model, metrics.accuracy)
        
        # 保存最佳模型
        if best_model:
            self._save_model(best_model[1], f"best_{best_model[0]}_classification.joblib")
        
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
        train_classes = set(y_train.tolist())
        test_classes = set(y_test.tolist())
        
        if train_classes != test_classes:
            raise ValueError(
                f"训练集和测试集类别不一致: "
                f"训练集 {sorted(train_classes)}, 测试集 {sorted(test_classes)}"
            )
    
    def _train_and_evaluate(
        self,
        name: str,
        model: Classifier,
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
        # 交叉验证
        cv_scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        
        # 全量训练
        model.fit(x_train, y_train)
        
        # 测试集预测
        y_pred = model.predict(x_test)
        
        # 计算指标
        metrics = MetricsCalculator.calculate(y_test, y_pred, average="macro")
        object.__setattr__(metrics, 'cv_accuracy', float(np.mean(cv_scores)))
        
        return metrics
    
    def _save_model(self, model: Classifier, filename: str) -> None:
        """保存模型到文件。
        
        Args:
            model: 分类器实例。
            filename: 文件名。
        """
        path = self.config.model_dir / filename
        joblib.dump(model, path)
    
    def _save_results(self, results: Dict[str, ClassificationMetrics]) -> None:
        """保存评估结果到 JSON 文件。
        
        Args:
            results: 模型名称到指标的映射。
        """
        output = {
            name: metrics.to_dict()
            for name, metrics in results.items()
        }
        
        best_name = max(results.keys(), key=lambda k: results[k].accuracy)
        output["best_model"] = best_name
        
        path = self.config.result_dir / "classification_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# 开集识别训练器
# ============================================================================


@dataclass(frozen=True)
class IdentificationModel:
    """开集识别模型。
    
    Attributes:
        known_labels: 已知身份标签列表。
        centroids: 各类别中心向量。
        threshold: 拒识距离阈值。
        distance_metric: 距离度量方式。
        unknown_label: 未知身份标签。
    """
    known_labels: List[str]
    centroids: np.ndarray
    threshold: float
    distance_metric: str = "euclidean"
    unknown_label: str = "Unknown"
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基于最小距离进行预测。
        
        Args:
            x: 输入特征，形状 (n_samples, n_features)。
            
        Returns:
            (predictions, min_distances) 元组。
        """
        # 计算到各类中心的欧氏距离
        distances = np.linalg.norm(
            x[:, None, :] - self.centroids[None, :, :], axis=2
        )
        
        min_dist = np.min(distances, axis=1)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_label = np.asarray(self.known_labels)[nearest_idx]
        
        predictions = np.asarray([
            label if dist <= self.threshold else self.unknown_label
            for label, dist in zip(nearest_label, min_dist)
        ], dtype=object)
        
        return predictions, min_dist


class IdentificationTrainer:
    """开集识别模型训练器。
    
    Attributes:
        config: 流水线配置对象。
        train_config: 训练配置。
        loader: 数据加载器。
    """
    
    def __init__(self, train_config: Optional[TrainingConfig] = None) -> None:
        """初始化识别训练器。
        
        Args:
            train_config: 训练配置，None 时使用默认配置。
        """
        self.config = PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.loader = DataLoader()
    
    def train(self, data_file: Optional[Path] = None) -> Dict[str, object]:
        """训练开集识别模型。
        
        Args:
            data_file: 预处理数据文件路径。
            
        Returns:
            包含模型和评估指标的字典。
        """
        # 加载数据
        data = self.loader.load_processed(data_file)
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=object)
        x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=object)
        
        # 获取已知身份
        known_subjects = set(y_train.tolist())
        
        # 映射测试标签（未知身份标记为 Unknown）
        y_test_open = np.asarray([
            label if label in known_subjects else "Unknown"
            for label in y_test.tolist()
        ], dtype=object)
        
        # 划分训练集和验证集（用于阈值估计）
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train,
            test_size=self.train_config.validation_split,
            random_state=self.train_config.random_seed,
            stratify=y_train,
        )
        
        # 计算类中心
        known_labels, centroids = self._compute_centroids(x_fit, y_fit)
        
        # 估计距离阈值
        threshold = self._estimate_threshold(x_val, y_val, known_labels, centroids)
        
        # 构建模型
        model = IdentificationModel(
            known_labels=[str(l) for l in known_labels.tolist()],
            centroids=centroids.astype(np.float32),
            threshold=threshold,
        )
        
        # 测试集预测
        y_pred, min_dist = model.predict(x_test)
        
        # 计算评估指标
        metrics = MetricsCalculator.calculate(y_test_open, y_pred, average="macro")
        
        # 保存模型
        self._save_model(model)
        
        # 构建结果
        results = {
            "metrics": metrics.to_dict(),
            "threshold": threshold,
            "known_subjects": sorted(known_subjects),
            "mean_min_distance": float(np.mean(min_dist)),
            "model": model,
        }
        
        self._save_results(results)
        
        return results
    
    def _compute_centroids(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算各类别的特征中心。
        
        Args:
            x: 特征矩阵。
            y: 标签数组。
            
        Returns:
            (labels, centroids) 元组。
        """
        unique_labels = np.asarray(sorted(set(y.tolist())), dtype=object)
        centroids = []
        
        for label in unique_labels:
            mask = y == label
            centroids.append(np.mean(x[mask], axis=0))
        
        return unique_labels, np.asarray(centroids, dtype=np.float32)
    
    def _estimate_threshold(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        known_labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """基于验证集估计拒识阈值。
        
        Args:
            x_val: 验证集特征。
            y_val: 验证集标签。
            known_labels: 已知标签列表。
            centroids: 类中心矩阵。
            
        Returns:
            距离阈值。
        """
        # 构建标签到索引的映射
        label_to_idx = {str(l): i for i, l in enumerate(known_labels.tolist())}
        
        # 获取每个样本对应类中心的索引
        true_indices = np.asarray([label_to_idx[str(y)] for y in y_val.tolist()])
        
        # 计算到真实类中心的距离
        distances = np.linalg.norm(x_val - centroids[true_indices], axis=1)
        
        # 使用分位数作为阈值
        return float(np.quantile(distances, self.train_config.distance_threshold_quantile))
    
    def _save_model(self, model: IdentificationModel) -> None:
        """保存模型到文件。
        
        Args:
            model: 识别模型实例。
        """
        path = self.config.model_dir / "identification_model.joblib"
        joblib.dump(model, path)
    
    def _save_results(self, results: Dict[str, object]) -> None:
        """保存评估结果到 JSON 文件。
        
        Args:
            results: 结果字典。
        """
        output = {
            "metrics": results["metrics"],
            "threshold": results["threshold"],
            "known_subjects": results["known_subjects"],
            "mean_min_distance": results["mean_min_distance"],
        }
        
        path = self.config.result_dir / "identification_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# 训练流水线
# ============================================================================


class TrainingPipeline:
    """模型训练流水线。
    
    整合分类和识别模型的训练流程。
    
    Attributes:
        config: 流水线配置对象。
        train_config: 训练配置。
        classification_trainer: 分类训练器。
        identification_trainer: 识别训练器。
    """
    
    def __init__(self, train_config: Optional[TrainingConfig] = None) -> None:
        """初始化训练流水线。
        
        Args:
            train_config: 训练配置。
        """
        self.config = PipelineConfig.from_root()
        self.train_config = train_config or TrainingConfig()
        self.classification_trainer = ClassificationTrainer(self.train_config)
        self.identification_trainer = IdentificationTrainer(self.train_config)
    
    def run_classification(self, data_file: Optional[Path] = None) -> dict:
        """运行分类模型训练。
        
        Args:
            data_file: 数据文件路径。
            
        Returns:
            分类结果字典。
        """
        results = self.classification_trainer.train(data_file)
        return {
            name: metrics.to_dict()
            for name, metrics in results.items()
        }
    
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


# ============================================================================
# 命令行入口
# ============================================================================


def main() -> None:
    """主函数：解析命令行参数并执行模型训练。"""
    parser = argparse.ArgumentParser(description="RSSI 模型训练工具")
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["classification", "identification", "both"],
        help="训练任务类型",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="交叉验证折数，默认 5",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="开集识别距离阈值分位数，默认 0.95",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
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


def _print_classification_results(results: dict) -> None:
    """打印分类模型结果。
    
    Args:
        results: 分类结果字典。
    """
    for name, metrics in results.items():
        if name == "best_model":
            continue
        print(f"\n{name}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        if 'cv_accuracy' in metrics:
            print(f"  CV准确率: {metrics['cv_accuracy']:.4f}")
    
    if "best_model" in results:
        print(f"\n最佳模型: {results['best_model']}")


def _print_identification_results(results: dict) -> None:
    """打印识别模型结果。
    
    Args:
        results: 识别结果字典。
    """
    metrics = results.get("metrics", {})
    print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"  精确率: {metrics.get('precision', 0):.4f}")
    print(f"  召回率: {metrics.get('recall', 0):.4f}")
    print(f"  F1分数: {metrics.get('f1', 0):.4f}")
    print(f"  拒识阈值: {results.get('threshold', 0):.4f}")
    print(f"  已知身份数: {len(results.get('known_subjects', []))}")


if __name__ == "__main__":
    main()