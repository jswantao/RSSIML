# -*- coding: utf-8 -*-
"""流水线配置模块。

提供数据处理流水线的配置管理功能，包括目录结构定义和默认参数配置。
使用 frozen dataclass 确保配置在运行时不可变，防止意外修改导致的难以排查的错误。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class PipelineConfig:
    """流水线配置数据类。

    存储数据处理流水线的所有配置参数。初始化后会自动生成子目录路径并尝试创建物理目录。

    Attributes:
        root_dir: 项目根目录路径。
        raw_dir: 原始数据存放目录 (只读属性，由 root_dir 衍生)。
        data_dir: 处理后数据存放目录 (只读属性，由 root_dir 衍生)。
        model_dir: 模型保存目录 (只读属性，由 root_dir 衍生)。
        result_dir: 结果输出目录 (只读属性，由 root_dir 衍生)。
        random_seed: 随机种子，用于结果复现。
        test_size:默认测试集分割比例。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        pca_variance: PCA保留方差比例。
    """

    # 类常量：定义子目录名称，便于统一管理和修改
    _SUB_DIRS: ClassVar[dict[str, str]] = {
        "raw_dir": "raw",
        "data_dir": "data",
        "model_dir": "models",
        "result_dir": "results",
    }

    # 实例字段
    root_dir: Path
    # 以下字段由 __post_init__ 自动计算，不在构造函数中直接接收
    raw_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    result_dir: Path = field(init=False)

    # 算法参数（带默认值）
    random_seed: int = 42
    test_size: float = 0.2
    window_size: int = 200
    step_size: int = 100
    pca_variance: float = 0.9019

    def __post_init__(self) -> None:
        """初始化后自动生成子目录路径并创建必要目录。

        由于 dataclass 是 frozen 的，必须使用 object.__setattr__ 来设置字段值。
        """
        for attr_name, sub_dir_name in self._SUB_DIRS.items():
            dir_path = self.root_dir / sub_dir_name
            # 绕过 frozen 限制设置属性
            object.__setattr__(self, attr_name, dir_path)
            # 确保目录存在
            self._ensure_dir(dir_path)

    @staticmethod
    def _ensure_dir(dir_path: Path) -> None:
        """确保目录存在（静默创建）。

        Args:
            dir_path: 需要确保存在的目录路径。
        """
        dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_root(cls, root_dir: Path | None = None) -> PipelineConfig:
        """从根目录创建默认配置。

        若未指定根目录，则自动推断为当前文件所在目录的上两级目录
        (假设 config.py 位于 project_root/scripts/)。

        Args:
            root_dir: 项目根目录路径。如果为 None，则自动推断。

        Returns:
            配置完成的 PipelineConfig 实例。
        """
        if root_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
        return cls(root_dir=root_dir)