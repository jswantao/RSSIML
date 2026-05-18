# -*- coding: utf-8 -*-
"""SVM 认证子包 — 批处理 SVC + 在线 SGD。"""
from scripts.models.svm.model import AuthenticationModel
from scripts.models.svm.trainers import SVMAuthenticationTrainer
from scripts.models.svm.utils import svm_scores
from scripts.models.svm.online import OnlineSVMVerifier, train_online_verifier

__all__ = [
    "AuthenticationModel",
    "SVMAuthenticationTrainer",
    "OnlineSVMVerifier",
    "train_online_verifier",
    "svm_scores",
]
