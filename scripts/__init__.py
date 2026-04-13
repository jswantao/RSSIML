# -*- coding: utf-8 -*-
"""RSSI 机器学习流水线脚本包。

该包提供了构建、运行和评估 RSSI 身份识别与分类模型的完整工具链，
包括数据加载、预处理、特征工程、模型训练和验证等核心模块。
"""

# --- 导入模块 ---

# 配置模块
from scripts.config import PipelineConfig

# 数据加载模块
from scripts.data_loader import DataFormatError, load_rssi_data

# 滑动窗口模块
from scripts.build_sliding_windows import (
    WindowBuilder,
    WindowConfig,
    WindowProcessor,
    build_sliding_windows,
    make_windows,
)

# 特征处理模块
from scripts.process_features_pca_norm import (
    FeatureConfig,
    FeatureExtractor,
    FeatureProcessor,
    extract_frequency_domain_features,
    process_features,
)

# 数据集划分模块
from scripts.split_rssi_dataset import (
    DatasetSplitter,
    Sample,
    split_classification_dataset,
    split_identification_dataset,
)

# 传统模型训练模块
from scripts.train_and_validate_models import (
    ClassificationMetrics,
    ClassificationTrainer,
    DataLoader,
    IdentificationModel,
    IdentificationTrainer,
    MetricsCalculator,
    ModelFactory,
    TrainingConfig,
    TrainingPipeline as TraditionalTrainingPipeline,  # 避免与 pipeline_runner 中的 TrainingPipeline 混淆
    train_classification,
    train_identification,
)

# CNN 模型训练模块
from scripts.train_cnn_models import (
    CNNCheckpoint,
    CNNConfig,
    CNNInference,
    CNNTrainer,
    RSSICNNClassifier,
    TrainingConfig as CNNTrainingConfig,  # 避免与上述 TrainingConfig 混淆
    load_cnn_checkpoint,
    predict_cnn_embeddings,
    predict_cnn_probabilities,
    predict_cnn_windows,
    train_cnn_classification,
    train_cnn_identification,
)

# 流水线运行模块
from scripts.pipeline_runner import (
    DataPipeline,
    PipelineParams,
    PipelineResult,
    TrainingPipeline,
    create_pipeline,
    run_classification_pipeline,
    run_identification_pipeline,
)

# --- 定义公共 API ---
__all__ = [
    # 配置
    "PipelineConfig",
    "WindowConfig",
    "FeatureConfig",
    "TrainingConfig",  # 传统模型的配置
    "CNNConfig",
    "CNNTrainingConfig",  # CNN 模型的配置
    "PipelineParams",
    "PipelineResult",
    # 异常
    "DataFormatError",
    # 数据加载与样本
    "load_rssi_data",
    "Sample",
    # 数据划分
    "DatasetSplitter",
    "split_classification_dataset",
    "split_identification_dataset",
    # 滑动窗口
    "WindowBuilder",
    "WindowProcessor",
    "build_sliding_windows",
    "make_windows",
    # 特征处理
    "FeatureExtractor",
    "FeatureProcessor",
    "extract_frequency_domain_features",
    "process_features",
    # 流水线
    "DataPipeline",
    "TrainingPipeline",  # pipeline_runner 中的高级流水线
    "TraditionalTrainingPipeline",  # train_and_validate_models 中的基础流水线
    "create_pipeline",
    "run_classification_pipeline",
    "run_identification_pipeline",
    # 传统模型
    "ClassificationMetrics",
    "ClassificationTrainer",
    "IdentificationModel",
    "IdentificationTrainer",
    "MetricsCalculator",
    "ModelFactory",
    "DataLoader",
    "train_classification",
    "train_identification",
    # CNN 模型
    "RSSICNNClassifier",
    "CNNTrainer",
    "CNNInference",
    "CNNCheckpoint",
    "load_cnn_checkpoint",
    "predict_cnn_windows",
    "predict_cnn_probabilities",
    "predict_cnn_embeddings",
    "train_cnn_classification",
    "train_cnn_identification",
]