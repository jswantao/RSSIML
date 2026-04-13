# -*- coding: utf-8 -*-
"""RSSI 数据处理与模型训练流水线模块。

提供分类任务和开集识别任务的端到端数据处理与模型训练流水线。
支持传统机器学习（SVM/RF/LR）和深度学习（1D-CNN）两种模型架构。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from scripts.build_sliding_windows import WindowConfig, WindowProcessor
from scripts.config import PipelineConfig as GlobalConfig
from scripts.process_features_pca_norm import FeatureConfig, FeatureProcessor
from scripts.split_rssi_dataset import DatasetSplitter
from scripts.train_and_validate_models import (
    TrainingConfig as TraditionalTrainingConfig,
    TrainingPipeline as TraditionalTrainingPipeline,
)
from scripts.train_cnn_models import (
    CNNTrainer,
    TrainingConfig as CNNTrainingConfig,
)

# 定义公共 API
__all__ = [
    "PipelineParams",
    "PipelineResult",
    "DataPipeline",
    "TrainingPipeline",
    "create_pipeline",
    "run_classification_pipeline",
    "run_identification_pipeline",
    "run_data_pipeline",
]

# 任务与模型类型定义
TaskType = Literal["classification", "identification"]
ModelType = Literal["traditional", "cnn"]


@dataclass(frozen=True)
class PipelineParams:
    """流水线参数配置。

    Attributes:
        test_size: 测试集分割比例。
        random_seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维（仅对传统 ML 有效）。
    """

    test_size: float = 0.2
    random_seed: int = 42
    window_size: int = 200
    step_size: int = 100
    pca_variance: float = 0.9019
    use_pca: bool = True

    def __post_init__(self) -> None:
        """验证参数有效性。

        Raises:
            ValueError: 当参数超出合法范围时抛出。
        """
        if not 0 < self.test_size < 0.5:
            raise ValueError(
                f"测试集比例应在 (0, 0.5) 之间，实际为 {self.test_size}"
            )
        if self.window_size < 2:
            raise ValueError(f"窗口大小必须大于 1，实际为 {self.window_size}")
        if self.step_size < 1:
            raise ValueError(f"步长必须大于 0，实际为 {self.step_size}")
        if not 0 < self.pca_variance <= 1:
            raise ValueError(
                f"PCA 方差比例应在 (0, 1] 之间，实际为 {self.pca_variance}"
            )


@dataclass
class PipelineResult:
    """流水线执行结果。

    Attributes:
        params: 流水线参数。
        split_meta: 数据划分元信息。
        window_meta: 滑动窗口元信息。
        process_meta: 特征处理元信息（仅传统 ML 有值）。
        split_file: 划分数据文件路径。
        window_file: 窗口数据文件路径。
        processed_file: 处理后数据文件路径（仅传统 ML 有值）。
        model_metrics: 模型评估指标。
        model_type: 使用的模型类型。
    """

    params: PipelineParams
    split_meta: dict
    window_meta: dict
    process_meta: Optional[dict]
    split_file: Path
    window_file: Path
    processed_file: Optional[Path]
    model_metrics: dict
    model_type: ModelType

    def to_dict(self) -> dict:
        """转换为字典格式。

        Returns:
            包含结果的字典。
        """
        result = {
            "model_type": self.model_type,
            "params": {
                "test_size": self.params.test_size,
                "random_seed": self.params.random_seed,
                "window_size": self.params.window_size,
                "step_size": self.params.step_size,
                "pca_variance": self.params.pca_variance,
                "use_pca": self.params.use_pca,
            },
            "split_meta": self.split_meta,
            "window_meta": self.window_meta,
            "split_file": str(self.split_file),
            "window_file": str(self.window_file),
        }
        if self.process_meta is not None:
            result["process_meta"] = self.process_meta
            result["processed_file"] = str(self.processed_file)
        
        result["model_metrics"] = self.model_metrics
        return result


class DataPipeline:
    """数据预处理流水线。

    整合数据加载、划分、窗口构建和特征处理的完整流程。
    根据任务类型生成不同的中间文件，避免冲突。

    Attributes:
        params: 流水线参数。
        global_config: 全局配置。
    """

    def __init__(self, params: Optional[PipelineParams] = None) -> None:
        """初始化数据流水线。

        Args:
            params: 流水线参数，None 时使用默认参数。
        """
        self.params = params or PipelineParams()
        self.global_config = GlobalConfig.from_root()

    def _get_task_suffix(self, task: TaskType) -> str:
        """获取任务对应的文件后缀。

        Args:
            task: 任务类型。

        Returns:
            文件后缀字符串。
        """
        return f"_{task}"

    def _get_default_paths(self, task: TaskType) -> tuple[Path, Path, Path]:
        """获取任务默认的中间文件路径。

        Args:
            task: 任务类型。

        Returns:
            (split_file, window_file, processed_file) 元组。
        """
        suffix = self._get_task_suffix(task)
        base_dir = self.global_config.data_dir

        split_file = base_dir / f"rssi_split{suffix}.pkl"
        window_file = base_dir / f"rssi_windowed{suffix}.pkl"
        processed_file = base_dir / f"rssi_processed{suffix}.pkl"

        return split_file, window_file, processed_file

    def run(self, task: TaskType) -> PipelineResult:
        """执行指定任务的数据流水线。

        Args:
            task: 任务类型 ("classification" 或 "identification")。

        Returns:
            PipelineResult 对象，包含所有中间文件路径和元信息。

        Raises:
            ValueError: 当任务类型非法时抛出。
        """
        if task not in ("classification", "identification"):
            raise ValueError(f"非法的任务类型: {task}")

        split_file, window_file, processed_file = self._get_default_paths(task)

        # 1. 数据划分
        splitter = DatasetSplitter(seed=self.params.random_seed)
        if task == "classification":
            split_result = splitter.split_classification(test_size=self.params.test_size)
        else:
            split_result = splitter.split_identification(test_size=self.params.test_size)

        # 保存划分结果到指定路径
        import pickle
        with split_file.open("wb") as f:
            pickle.dump(split_result, f)

        # 2. 滑动窗口构建
        window_processor = WindowProcessor(
            WindowConfig(
                window_size=self.params.window_size,
                step_size=self.params.step_size,
            ),
            config=self.global_config,
        )
        # WindowProcessor 内部会保存，但我们这里显式指定输出路径以确保一致性
        window_result = window_processor.process(
            split_file=split_file,
            output_file=window_file
        )

        # 3. 特征处理 (仅用于传统 ML，但为了流水线完整性，我们总是生成)
        # 注意：CNN 训练器将直接使用 window_file，忽略 processed_file
        feature_processor = FeatureProcessor(
            FeatureConfig(
                use_pca=self.params.use_pca,
                pca_variance=self.params.pca_variance,
            ),
            config=self.global_config,
        )
        process_result = feature_processor.process(
            input_file=window_file,
            output_file=processed_file
        )

        return PipelineResult(
            params=self.params,
            split_meta=split_result["meta"],
            window_meta=window_result["meta"],
            process_meta=process_result["meta"],
            split_file=split_file,
            window_file=window_file,
            processed_file=processed_file,
            model_metrics={},  # 占位符
            model_type="traditional",  # 占位符，将在 TrainingPipeline 中更新
        )


class TrainingPipeline:
    """模型训练流水线。

    整合数据预处理和模型训练的完整流程。
    支持传统机器学习和 CNN 两种模型架构。

    Attributes:
        params: 流水线参数。
        data_pipeline: 数据预处理流水线。
        traditional_config: 传统模型训练配置。
        cnn_config: CNN 模型训练配置。
    """

    def __init__(
        self,
        params: Optional[PipelineParams] = None,
        traditional_config: Optional[TraditionalTrainingConfig] = None,
        cnn_config: Optional[CNNTrainingConfig] = None,
    ) -> None:
        """初始化训练流水线。

        Args:
            params: 流水线参数。
            traditional_config: 传统模型训练配置。
            cnn_config: CNN 模型训练配置。
        """
        self.params = params or PipelineParams()
        self.data_pipeline = DataPipeline(self.params)
        self.traditional_config = traditional_config or TraditionalTrainingConfig(
            random_seed=self.params.random_seed
        )
        self.cnn_config = cnn_config or CNNTrainingConfig(
            random_seed=self.params.random_seed
        )

    def run(
        self,
        task: TaskType,
        model_type: ModelType = "traditional",
    ) -> PipelineResult:
        """执行指定任务和模型类型的完整流水线。

        Args:
            task: 任务类型 ("classification" 或 "identification")。
            model_type: 模型类型 ("traditional" 或 "cnn")。

        Returns:
            PipelineResult 对象，包含数据信息和模型评估指标。
        """
        # 1. 数据预处理
        data_result = self.data_pipeline.run(task)

        # 2. 模型训练
        if model_type == "traditional":
            metrics = self._train_traditional(task, data_result.processed_file)
        elif model_type == "cnn":
            # CNN 通常直接使用窗口数据，因为需要保留时序结构
            # 对于分类任务，CNN 也可以接受处理后特征，但通常效果不如原始窗口
            # 这里我们约定：CNN 使用 window_file
            metrics = self._train_cnn(task, data_result.window_file)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 更新结果
        data_result.model_metrics = metrics
        data_result.model_type = model_type
        
        # 如果是 CNN，process_meta 和 processed_file 可能不适用，设为 None 或保留
        if model_type == "cnn":
            data_result.process_meta = None
            data_result.processed_file = None

        return data_result

    def _train_traditional(self, task: TaskType, data_file: Optional[Path]) -> dict:
        """训练传统机器学习模型。

        Args:
            task: 任务类型。
            data_file: 预处理后的数据文件路径。

        Returns:
            模型评估指标字典。
        """
        pipeline = TraditionalTrainingPipeline(
            train_config=self.traditional_config,
            config=self.data_pipeline.global_config,
        )

        if task == "classification":
            return pipeline.run_classification(data_file=data_file)
        else:
            return pipeline.run_identification(data_file=data_file)

    def _train_cnn(self, task: TaskType, data_file: Optional[Path]) -> dict:
        """训练 CNN 模型。

        Args:
            task: 任务类型。
            data_file: 滑动窗口数据文件路径。

        Returns:
            模型评估指标字典。
        """
        trainer = CNNTrainer(
            train_config=self.cnn_config,
            config=self.data_pipeline.global_config,
        )

        if task == "classification":
            result = trainer.train_classification(data_file=data_file)
            # 提取关键指标以便统一格式
            return result.get("classification", {})
        else:
            result = trainer.train_identification(data_file=data_file)
            # 提取关键指标
            return result.get("identification", {})

    def run_classification(
        self, model_type: ModelType = "traditional"
    ) -> PipelineResult:
        """执行分类任务完整流水线。

        Args:
            model_type: 模型类型。

        Returns:
            PipelineResult 对象。
        """
        return self.run("classification", model_type)

    def run_identification(
        self, model_type: ModelType = "traditional"
    ) -> PipelineResult:
        """执行识别任务完整流水线。

        Args:
            model_type: 模型类型。

        Returns:
            PipelineResult 对象。
        """
        return self.run("identification", model_type)


# =============================================================================
# 便捷函数
# =============================================================================


def create_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
) -> TrainingPipeline:
    """创建训练流水线实例。

    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。

    Returns:
        TrainingPipeline 实例。
    """
    params = PipelineParams(
        test_size=test_size,
        random_seed=seed,
        window_size=window_size,
        step_size=step_size,
        pca_variance=pca_variance,
        use_pca=use_pca,
    )
    return TrainingPipeline(params)


def run_classification_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
    model_type: ModelType = "traditional",
) -> dict:
    """执行分类任务完整流水线（便捷函数）。

    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        model_type: 模型类型 ("traditional" 或 "cnn")。

    Returns:
        包含流水线执行结果的字典。
    """
    pipeline = create_pipeline(
        test_size=test_size,
        seed=seed,
        window_size=window_size,
        step_size=step_size,
        pca_variance=pca_variance,
        use_pca=use_pca,
    )
    result = pipeline.run_classification(model_type=model_type)
    return result.to_dict()


def run_identification_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
    model_type: ModelType = "traditional",
) -> dict:
    """执行识别任务完整流水线（便捷函数）。

    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        model_type: 模型类型 ("traditional" 或 "cnn")。

    Returns:
        包含流水线执行结果的字典。
    """
    pipeline = create_pipeline(
        test_size=test_size,
        seed=seed,
        window_size=window_size,
        step_size=step_size,
        pca_variance=pca_variance,
        use_pca=use_pca,
    )
    result = pipeline.run_identification(model_type=model_type)
    return result.to_dict()


def run_data_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
    task: TaskType = "classification",
) -> dict:
    """执行数据预处理流水线（不包含模型训练）。

    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        task: 任务类型。

    Returns:
        包含数据预处理结果的字典。
    """
    params = PipelineParams(
        test_size=test_size,
        random_seed=seed,
        window_size=window_size,
        step_size=step_size,
        pca_variance=pca_variance,
        use_pca=use_pca,
    )
    pipeline = DataPipeline(params)
    result = pipeline.run(task)
    
    # 转换为字典，移除模型相关字段
    res_dict = result.to_dict()
    res_dict.pop("model_metrics", None)
    res_dict.pop("model_type", None)
    return res_dict


# =============================================================================
# 命令行入口
# =============================================================================


def main() -> None:
    """主函数：解析命令行参数并执行流水线。"""
    parser = argparse.ArgumentParser(description="RSSI 机器学习流水线工具")
    
    # 通用参数
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "identification"],
        help="任务类型",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="traditional",
        choices=["traditional", "cnn"],
        help="模型类型",
    )
    
    # 数据参数
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--window-size", type=int, default=200, help="窗口大小")
    parser.add_argument("--step-size", type=int, default=100, help="滑动步长")
    parser.add_argument("--pca-variance", type=float, default=0.9019, help="PCA 方差比例")
    parser.add_argument("--use-pca", action="store_true", default=True, help="启用 PCA")
    parser.add_argument("--disable-pca", action="store_true", help="禁用 PCA")

    # 传统模型参数
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--threshold-quantile", type=float, default=0.95, help="开集阈值分位数")

    # CNN 模型参数
    parser.add_argument("--epochs", type=int, default=20, help="CNN 训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="CNN 批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="CNN 学习率")
    
    args = parser.parse_args()

    # 创建流水线
    params = PipelineParams(
        test_size=args.test_size,
        random_seed=args.seed,
        window_size=args.window_size,
        step_size=args.step_size,
        pca_variance=args.pca_variance,
        use_pca=not args.disable_pca,
    )

    trad_config = TraditionalTrainingConfig(
        cv_folds=args.cv_folds,
        distance_threshold_quantile=args.threshold_quantile,
        random_seed=args.seed,
    )

    cnn_config = CNNTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
    )

    pipeline = TrainingPipeline(
        params=params,
        traditional_config=trad_config,
        cnn_config=cnn_config,
    )

    print(f"🚀 开始执行流水线: Task={args.task}, Model={args.model_type}")
    

if __name__ == "__main__":
    main()