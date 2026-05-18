# -*- coding: utf-8 -*-
"""WiFiML — 基于 Wi-Fi 信号的身份认证流水线。

提供从原始数据加载到模型训练和在线推理的完整工具链，
支持 SVM 和 1D-CNN 两条建模路线，RSSI (MAT) 和 CSI (NPY) 双数据源。
"""
import logging

# --- 导入模块 ---

# 配置模块
from scripts.config import PipelineConfig

# 数据加载模块
from scripts.data_loader import (
    DataFormatError,
    DataLoadContext,
    estimate_matrix_memory,
    load_csi_data,
    load_csi_data_generator,
    load_npy_matrix,
    load_rssi_data,
    load_rssi_data_generator,
    parse_npy_filename,
    validate_matrix,
)

# 滑动窗口模块
from scripts.build_sliding_windows import (
    WindowBuilder,
    WindowConfig,
    WindowProcessor,
)

# 特征处理模块
from scripts.process_features_pca_norm import (
    FeatureExtractor,
    FeatureProcessor,
    PreprocessConfig,
)

# 数据集划分模块
from scripts.split_rssi_dataset import (
    DatasetSplitter,
    DatasetSummary,
    Sample,
    split_authentication_dataset,
)

# 统一模型训练模块
from scripts.models import (
    # Configs
    CNNConfig,
    CNNTrainConfig,
    SVMConfig,
    # Models
    AuthenticationModel,
    CNNAuthenticationModel,
    RSSICNNBinaryClassifier,
    # Trainers
    CNNInference,
    CNNTrainer,
    SVMAuthenticationTrainer,
    TrainingError,
    # Functions
    MetricsCalculator,
    compute_threshold,
    evaluate_authentication,
    svm_scores,
    load_cnn_checkpoint,
    train_cnn_authentication,
)

# 流水线运行模块
from scripts.pipeline_runner import (
    AuthPipeline,
    DataPipeline,
    PipelineParams,
    PipelineResult,
    run_authentication_pipeline,
    run_data_pipeline,
    run_npy_authentication_cnn,
    run_npy_authentication_svm,
)

# --- 定义公共 API ---
__all__ = [
    # 配置
    "PipelineConfig",
    "WindowConfig",
    "PreprocessConfig",
    "SVMConfig",
    "CNNConfig",
    "CNNTrainConfig",
    "PipelineParams",
    "PipelineResult",
    # 异常
    "DataFormatError",
    "DataLoadContext",
    "TrainingError",
    # 数据加载
    "estimate_matrix_memory",
    "load_rssi_data",
    "load_rssi_data_generator",
    "load_csi_data",
    "load_csi_data_generator",
    "load_npy_matrix",
    "parse_npy_filename",
    "validate_matrix",
    "Sample",
    # 数据摘要
    "DatasetSummary",
    # 数据划分
    "DatasetSplitter",
    "split_authentication_dataset",
    # 滑动窗口
    "WindowBuilder",
    "WindowProcessor",
    # 特征处理
    "FeatureExtractor",
    "FeatureProcessor",
    # 流水线
    "DataPipeline",
    "AuthPipeline",
    "run_authentication_pipeline",
    "run_data_pipeline",
    "run_npy_authentication_svm",
    "run_npy_authentication_cnn",
    # SVM 模型
    "AuthenticationModel",
    "SVMAuthenticationTrainer",
    "svm_scores",
    # CNN 模型
    "RSSICNNBinaryClassifier",
    "CNNAuthenticationModel",
    "CNNTrainer",
    "CNNInference",
    "load_cnn_checkpoint",
    "train_cnn_authentication",
    # 共享函数
    "MetricsCalculator",
    "compute_threshold",
    "evaluate_authentication",
]

# 包级日志配置
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
