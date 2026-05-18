# -*- coding: utf-8 -*-
"""SVM 认证模型容器。"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from scripts.models import MODEL_VERSION  # 延迟: __init__.py 先定义 MODEL_VERSION


class AuthenticationModel:
    """身份认证模型 (SVM) — 支持跨进程 pickle 序列化和特征维度自适应。"""

    def __init__(
        self,
        verifiers: dict,
        thresholds: dict[str, float],
        label_encoder: LabelEncoder,
        subjects: list[str] | None = None,
        threshold_method: str = "youden",
    ):
        self.verifiers = verifiers
        self.thresholds = thresholds
        self.label_encoder = label_encoder
        self.subjects = subjects or list(verifiers.keys())
        self.threshold_method = threshold_method
        self.pca_model = None
        self.scaler_model = None
        self.feature_config: dict | None = None  # 推理时自适应特征提取
        self.feature_dim: int | None = None  # 训练时特征维度, 推理校验
        self.data_source: str | None = None  # "rssi" 或 "csi", 推理时校验
        self.model_version = MODEL_VERSION

    def __reduce__(self):
        return (
            AuthenticationModel,
            (self.verifiers, self.thresholds, self.label_encoder,
             self.subjects, self.threshold_method),
            {"pca_model": self.pca_model, "scaler_model": self.scaler_model,
             "feature_config": self.feature_config,
             "feature_dim": self.feature_dim,
             "data_source": self.data_source,
             "model_version": self.model_version},
        )

    def __setstate__(self, state):
        self.pca_model = state.get("pca_model")
        self.scaler_model = state.get("scaler_model")
        self.feature_config = state.get("feature_config")
        self.feature_dim = state.get("feature_dim")
        self.data_source = state.get("data_source")
        self.model_version = state.get("model_version", MODEL_VERSION)
