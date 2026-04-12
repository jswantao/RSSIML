"""RSSI 数据处理流水线模块。

提供分类任务和开集识别任务的端到端数据处理与模型训练流水线。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


from scripts.config import PipelineConfig as GlobalConfig
from scripts.split_rssi_dataset import DatasetSplitter
from scripts.build_sliding_windows import WindowConfig, WindowProcessor
from scripts.process_features_pca_norm import FeatureConfig, FeatureProcessor
from scripts.train_and_validate_models import TrainingPipeline as ModelTrainingPipeline



# ============================================================================
# 配置定义
# ============================================================================


@dataclass(frozen=True)
class PipelineParams:
    """流水线参数配置。
    
    Attributes:
        test_size: 测试集比例。
        random_seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        low_freq_bins: 低频幅值频点数。
    """
    test_size: float = 0.2
    random_seed: int = 42
    window_size: int = 200
    step_size: int = 100
    pca_variance: float = 0.9019
    use_pca: bool = True
    low_freq_bins: int = 16
    
    def __post_init__(self) -> None:
        """验证配置参数的有效性。"""
        if not 0 < self.test_size < 0.5:
            raise ValueError(
                f"测试集比例应在 (0, 0.5) 之间，实际为 {self.test_size}"
            )
        if self.window_size < 2:
            raise ValueError(
                f"窗口大小必须大于 1，实际为 {self.window_size}"
            )
        if self.step_size < 1:
            raise ValueError(
                f"步长必须大于 0，实际为 {self.step_size}"
            )
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
        process_meta: 特征处理元信息。
        split_file: 划分数据文件路径。
        window_file: 窗口数据文件路径。
        processed_file: 处理后数据文件路径。
        model_metrics: 模型评估指标。
    """
    params: PipelineParams
    split_meta: dict
    window_meta: dict
    process_meta: dict
    split_file: Path
    window_file: Path
    processed_file: Path
    model_metrics: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """转换为字典格式。"""
        result = {
            "params": {
                "test_size": self.params.test_size,
                "random_seed": self.params.random_seed,
                "window_size": self.params.window_size,
                "step_size": self.params.step_size,
                "pca_variance": self.params.pca_variance,
                "use_pca": self.params.use_pca,
                "low_freq_bins": self.params.low_freq_bins,
            },
            "split_meta": self.split_meta,
            "window_meta": self.window_meta,
            "process_meta": self.process_meta,
            "split_file": str(self.split_file),
            "window_file": str(self.window_file),
            "processed_file": str(self.processed_file),
        }
        if self.model_metrics is not None:
            result["model_metrics"] = self.model_metrics
        return result


# ============================================================================
# 数据流水线
# ============================================================================


class DataPipeline:
    """数据预处理流水线。
    
    整合数据加载、划分、窗口构建和特征处理的完整流程。
    
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
    
    def run_classification(self) -> PipelineResult:
        """执行分类任务数据流水线。
        
        为身份分类任务准备训练和测试数据，确保每个受试者的数据
        在训练集和测试集中都有分布。
        
        Returns:
            PipelineResult 对象，包含所有中间文件路径和元信息。
        """
        splitter = DatasetSplitter(seed=self.params.random_seed)

        # 1. 数据划分
        split_result = splitter.split_classification(test_size=self.params.test_size)
        split_file = self.global_config.data_dir / "rssi_split_classification.pkl"
        
        # 2. 滑动窗口构建
        window_processor = WindowProcessor(
            WindowConfig(
                window_size=self.params.window_size,
                step_size=self.params.step_size,
            )
        )
        window_result = window_processor.process(
            split_file=split_file,
            output_file=self.global_config.data_dir / "rssi_windowed_classification.pkl",
        )
        window_file = self.global_config.data_dir / "rssi_windowed_classification.pkl"
        
        # 3. 特征处理
        feature_processor = FeatureProcessor(
            FeatureConfig(
                low_freq_bins=self.params.low_freq_bins,
                use_pca=self.params.use_pca,
                pca_variance=self.params.pca_variance,
            )
        )
        process_result = feature_processor.process(
            input_file=window_file,
            output_file=self.global_config.data_dir / "rssi_processed_classification.pkl",
        )
        processed_file = self.global_config.data_dir / "rssi_processed_classification.pkl"
        
        return PipelineResult(
            params=self.params,
            split_meta=split_result.get("meta", {}),
            window_meta=window_result.get("meta", {}),
            process_meta=process_result.get("meta", {}),
            split_file=split_file,
            window_file=window_file,
            processed_file=processed_file,
        )
    
    def run_identification(self) -> PipelineResult:
        """执行识别任务数据流水线。
        
        为开集识别任务准备数据，确保训练集和测试集的受试者不重叠。
        
        Returns:
            PipelineResult 对象，包含所有中间文件路径和元信息。
        """
        splitter = DatasetSplitter(seed=self.params.random_seed)

        # 1. 数据划分（人员分离）
        split_result = splitter.split_identification(test_size=self.params.test_size)
        split_file = self.global_config.data_dir / "rssi_split_identification.pkl"
        
        # 2. 滑动窗口构建
        window_processor = WindowProcessor(
            WindowConfig(
                window_size=self.params.window_size,
                step_size=self.params.step_size,
            )
        )
        window_result = window_processor.process(
            split_file=split_file,
            output_file=self.global_config.data_dir / "rssi_windowed_identification.pkl",
        )
        window_file = self.global_config.data_dir / "rssi_windowed_identification.pkl"
        
        # 3. 特征处理
        feature_processor = FeatureProcessor(
            FeatureConfig(
                low_freq_bins=self.params.low_freq_bins,
                use_pca=self.params.use_pca,
                pca_variance=self.params.pca_variance,
            )
        )
        process_result = feature_processor.process(
            input_file=window_file,
            output_file=self.global_config.data_dir / "rssi_processed_identification.pkl",
        )
        processed_file = self.global_config.data_dir / "rssi_processed_identification.pkl"
        
        return PipelineResult(
            params=self.params,
            split_meta=split_result.get("meta", {}),
            window_meta=window_result.get("meta", {}),
            process_meta=process_result.get("meta", {}),
            split_file=split_file,
            window_file=window_file,
            processed_file=processed_file,
        )


# ============================================================================
# 训练流水线
# ============================================================================


class TrainingPipeline:
    """模型训练流水线。
    
    整合数据处理和模型训练的完整流程。
    
    Attributes:
        params: 流水线参数。
        data_pipeline: 数据预处理流水线。
    """
    
    def __init__(self, params: Optional[PipelineParams] = None) -> None:
        """初始化训练流水线。
        
        Args:
            params: 流水线参数，None 时使用默认参数。
        """
        self.params = params or PipelineParams()
        self.data_pipeline = DataPipeline(self.params)
    
    def run_classification(self) -> PipelineResult:
        """执行分类任务完整流水线。
        
        包括数据预处理和分类模型训练。
        
        Returns:
            PipelineResult 对象，包含数据信息和模型评估指标。
        """
        # 数据预处理
        result = self.data_pipeline.run_classification()
        
        # 模型训练
        model_result = ModelTrainingPipeline().run_classification(
            data_file=result.processed_file
        )
        result.model_metrics = model_result
        
        return result
    
    def run_identification(self, target_subject: Optional[str] = None) -> PipelineResult:
        """执行识别任务完整流水线。
        
        包括数据预处理和开集识别模型训练。
        
        Args:
            target_subject: 目标受试者标识符（可选）。
            
        Returns:
            PipelineResult 对象，包含数据信息和模型评估指标。
        """
        # 数据预处理
        result = self.data_pipeline.run_identification()
        
        # 模型训练
        model_result = ModelTrainingPipeline().run_identification(
            data_file=result.processed_file,
        )
        result.model_metrics = model_result
        
        return result


# ============================================================================
# 便捷函数
# ============================================================================


def create_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
    low_freq_bins: int = 16,
) -> TrainingPipeline:
    """创建训练流水线实例。
    
    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        low_freq_bins: 低频幅值频点数。
        
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
        low_freq_bins=low_freq_bins,
    )
    return TrainingPipeline(params)


def run_classification_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
) -> dict:
    """执行分类任务完整流水线（便捷函数）。
    
    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        
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
    result = pipeline.run_classification()
    return result.to_dict()


def run_identification_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
    target_subject: Optional[str] = None,
) -> dict:
    """执行识别任务完整流水线（便捷函数）。
    
    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        target_subject: 目标受试者标识符（可选）。
        
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
    result = pipeline.run_identification(target_subject=target_subject)
    return result.to_dict()


def run_classification_data_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
) -> dict:
    """执行分类任务数据流水线（仅数据预处理）。
    
    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        
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
    result = pipeline.run_classification()
    return result.to_dict()


def run_identification_data_pipeline(
    test_size: float = 0.2,
    seed: int = 42,
    window_size: int = 200,
    step_size: int = 100,
    pca_variance: float = 0.9019,
    use_pca: bool = True,
) -> dict:
    """执行识别任务数据流水线（仅数据预处理）。
    
    Args:
        test_size: 测试集比例。
        seed: 随机种子。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA 保留方差比例。
        use_pca: 是否使用 PCA 降维。
        
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
    result = pipeline.run_identification()
    return result.to_dict()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例 1：使用类接口
    print("=" * 50)
    print("示例 1：使用类接口执行分类流水线")
    print("=" * 50)
    
    params = PipelineParams(
        test_size=0.2,
        random_seed=42,
        window_size=200,
        step_size=100,
        pca_variance=0.95,
        use_pca=True,
    )
    
    pipeline = TrainingPipeline(params)
    result = pipeline.run_classification()
    
    print(f"数据划分: {result.split_meta.get('num_train_files', 'N/A')} 训练文件, "
          f"{result.split_meta.get('num_test_files', 'N/A')} 测试文件")
    print(f"滑动窗口: {result.window_meta.get('num_total_windows', 'N/A')} 个窗口")
    print(f"特征处理: PCA 组件数 = {result.process_meta.get('pca_components', 'N/A')}")
    
    if result.model_metrics:
        print(f"模型指标: {result.model_metrics}")
    
    # 示例 2：使用便捷函数
    print("\n" + "=" * 50)
    print("示例 2：使用便捷函数执行识别流水线")
    print("=" * 50)
    
    result_dict = run_identification_pipeline(
        test_size=0.2,
        seed=42,
        window_size=150,
        step_size=75,
    )
    
    print(f"参数: {result_dict.get('params', {})}")
    print(f"模型指标: {result_dict.get('model_metrics', {})}")