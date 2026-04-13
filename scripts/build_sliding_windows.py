# -*- coding: utf-8 -*-
"""RSSI 滑动窗口构建模块。

提供将时序 RSSI 数据转换为滑动窗口样本的功能。
采用高效的 NumPy  stride tricks 实现零拷贝或低拷贝窗口化，适用于大规模时间序列数据预处理。
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.config import PipelineConfig

# 定义公共 API
__all__ = [
    "WindowConfig",
    "WindowBuilder",
    "WindowProcessor",
    "make_windows",
    "build_sliding_windows",
]


@dataclass(frozen=True)
class WindowConfig:
    """滑动窗口配置。

    Attributes:
        window_size: 窗口长度（时间步数）。
        step_size: 滑动步长（时间步数）。
    """

    window_size: int = 200
    step_size: int = 100

    def __post_init__(self) -> None:
        """验证配置参数的有效性。

        Raises:
            ValueError: 当窗口大小 < 2 或步长 < 1 时抛出。
        """
        if self.window_size < 2:
            raise ValueError(f"窗口大小必须大于 1，实际为 {self.window_size}")
        if self.step_size < 1:
            raise ValueError(f"步长必须大于 0，实际为 {self.step_size}")


class WindowBuilder:
    """滑动窗口构建器。

    利用 NumPy 高级索引技巧高效地将二维时序数据转换为三维窗口数据。

    Attributes:
        config: 流水线全局配置（用于获取默认路径等，可选）。
        window_config: 滑动窗口具体配置。
    """

    def __init__(
        self,
        window_config: Optional[WindowConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化窗口构建器。

        Args:
            window_config: 滑动窗口配置。若为 None，则使用默认配置。
            config: 流水线全局配置。若为 None，则在需要时自动推断。
        """
        self.window_config = window_config or WindowConfig()
        self._config = config

    @property
    def config(self) -> PipelineConfig:
        """懒加载全局配置。"""
        if self._config is None:
            self._config = PipelineConfig.from_root()
        return self._config

    def build(self, data: np.ndarray) -> np.ndarray:
        """将二维时序数据转换为滑动窗口样本。

        使用 np.lib.stride_tricks.sliding_window_view 实现高效窗口化。
        如果数据长度不足，返回形状为 (0, window_size, n_channels) 的空数组。

        Args:
            data: 二维数组，形状 (length, n_channels)。

        Returns:
            三维数组，形状 (n_windows, window_size, n_channels)。
            数据类型为 float32。

        Raises:
            ValueError: 当输入数据维度不为 2 时抛出。
        """
        if data.ndim != 2:
            raise ValueError(f"输入数据必须是二维数组 (L, C)，实际形状: {data.shape}")

        length, n_channels = data.shape
        window_size = self.window_config.window_size
        step_size = self.window_config.step_size

        # 边界情况：数据长度不足以形成一个窗口
        if length < window_size:
            return np.empty((0, window_size, n_channels), dtype=np.float32)

        # 确保数据是 C-contiguous 以优化 stride 操作
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            data = data.astype(np.float32, copy=False)

        try:
            # NumPy 1.20+ 推荐方式
            from numpy.lib.stride_tricks import sliding_window_view
            
            # sliding_window_view 返回视图，我们需要复制以确保后续操作安全且内存连续
            # window_shape=(window_size, n_channels) 会在最后两个轴上滑动
            # 但我们的数据是 (L, C)，我们只想在 L 轴滑动，保持 C 轴完整
            # 因此我们对 axis=0 进行滑动
            windows = sliding_window_view(data, window_shape=window_size, axis=0)
            
            # 此时形状为 (L - window_size + 1, window_size, n_channels)
            # 我们需要应用 step_size 进行采样
            windows = windows[::step_size]
            # 实际上 axis=0 滑窗后维度为 (N, C, W)，需转为 (N, W, C)
            windows = np.transpose(windows, (0, 2, 1))
            
        except ImportError:
            # 兼容旧版本 NumPy 的手动实现（较少用到，但作为后备）
            starts = np.arange(0, length - window_size + 1, step_size)
            indices = np.arange(window_size)
            # 使用高级索引生成窗口
            windows = data[starts[:, None] + indices[None, :]]

        # 确保返回的是 float32 且内存连续
        return np.ascontiguousarray(windows, dtype=np.float32)

    def build_from_samples(
        self, samples: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """从样本列表批量构建滑动窗口。

        Args:
            samples: 样本字典列表。每个字典应包含:
                - 'data': np.ndarray, 形状 (L, C)
                - 'subject': str, 受试者标识
                - 'file_name': str, 文件名

        Returns:
            元组 (features, labels, source_files):
                - features: np.ndarray, 形状 (N_total_windows, window_size, n_channels)
                - labels: np.ndarray, 形状 (N_total_windows,), dtype=object
                - source_files: List[str], 每个窗口对应的源文件名
        """
        features_list: List[np.ndarray] = []
        labels_list: List[str] = []
        files_list: List[str] = []

        # 预分配估算（可选优化，此处暂略，直接 append）
        for sample in samples:
            raw_data = sample.get("data")
            if raw_data is None:
                continue
            
            # 确保数据类型和形状
            try:
                data = np.asarray(raw_data, dtype=np.float32)
                if data.ndim != 2:
                    # 尝试重塑或跳过
                    continue
            except Exception:
                continue

            windows = self.build(data)

            # 如果该样本产生的窗口数为 0，跳过
            if windows.shape[0] == 0:
                continue

            n_windows = windows.shape[0]
            features_list.append(windows)
            
            subject_id = str(sample.get("subject", "unknown"))
            file_name = str(sample.get("file_name", "unknown.mat"))
            
            labels_list.extend([subject_id] * n_windows)
            files_list.extend([file_name] * n_windows)

        # 合并结果
        if not features_list:
            # 返回空结构，保持维度一致性
            w_size = self.window_config.window_size
            # 尝试从第一个样本推断通道数，否则默认为 0
            n_channels = 0
            if samples and "data" in samples[0]:
                try:
                    d = np.asarray(samples[0]["data"], dtype=np.float32)
                    if d.ndim == 2:
                        n_channels = d.shape[1]
                except Exception:
                    pass
            return (
                np.empty((0, w_size, n_channels), dtype=np.float32),
                np.array([], dtype=object),
                [],
            )

        final_features = np.concatenate(features_list, axis=0)
        final_labels = np.array(labels_list, dtype=object)
        
        return final_features, final_labels, files_list


class WindowProcessor:
    """滑动窗口处理流水线。

    整合数据加载、窗口构建和结果持久化。

    Attributes:
        builder: 窗口构建器实例。
        config: 流水线全局配置。
    """

    def __init__(
        self,
        window_config: Optional[WindowConfig] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化窗口处理器。

        Args:
            window_config: 滑动窗口配置。
            config: 流水线全局配置。
        """
        self.config = config or PipelineConfig.from_root()
        self.builder = WindowBuilder(window_config=window_config, config=self.config)

    def process(
        self,
        split_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict:
        """读取划分数据并生成滑动窗口样本。

        Args:
            split_file: 输入划分数据文件 (.pkl)。若为 None，使用默认路径。
            output_file: 输出文件路径 (.pkl)。若为 None，使用默认路径。

        Returns:
            包含窗口化数据和元信息的字典。

        Raises:
            FileNotFoundError: 当输入文件不存在时抛出。
            KeyError: 当输入数据缺少 'train' 或 'test' 键时抛出。
        """
        input_path = split_file or (self.config.data_dir / "rssi_split_classification.pkl")
        
        # 备选：如果分类划分文件不存在，尝试识别划分文件
        if not input_path.exists():
            alt_path = self.config.data_dir / "rssi_split_identification.pkl"
            if alt_path.exists():
                input_path = alt_path
            else:
                raise FileNotFoundError(f"未找到划分文件: {input_path} 或 {alt_path}")

        with input_path.open("rb") as f:
            split_data = pickle.load(f)

        if "train" not in split_data or "test" not in split_data:
            raise KeyError("划分数据必须包含 'train' 和 'test' 键")

        # 构建训练集窗口
        x_train, y_train, train_files = self.builder.build_from_samples(
            split_data["train"]
        )
        
        # 构建测试集窗口
        x_test, y_test, test_files = self.builder.build_from_samples(
            split_data["test"]
        )

        meta = {
            "window_size": self.builder.window_config.window_size,
            "step_size": self.builder.window_config.step_size,
            "num_train_windows": int(x_train.shape[0]),
            "num_test_windows": int(x_test.shape[0]),
            "num_total_windows": int(x_train.shape[0] + x_test.shape[0]),
            "train_feature_shape": list(x_train.shape),
            "test_feature_shape": list(x_test.shape),
            "source_split_file": str(input_path.name),
        }

        output = {
            "meta": meta,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "train_source_files": train_files,
            "test_source_files": test_files,
        }

        # 保存结果
        out_path = output_file or (self.config.data_dir / "rssi_windowed.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_path.open("wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        return output


# =============================================================================
# 便捷函数
# =============================================================================

def make_windows(
    matrix: np.ndarray, window_size: int = 200, step_size: int = 100
) -> np.ndarray:
    """便捷函数：从单个二维矩阵生成滑动窗口。

    Args:
        matrix: 输入二维时序矩阵 (L, C)。
        window_size: 窗口大小。
        step_size: 滑动步长。

    Returns:
        三维滑窗数组 (N, W, C)。
    """
    config = WindowConfig(window_size=window_size, step_size=step_size)
    builder = WindowBuilder(window_config=config)
    return builder.build(matrix)


def build_sliding_windows(
    split_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    window_size: int = 200,
    step_size: int = 100,
    config: Optional[PipelineConfig] = None,
) -> dict:
    """便捷函数：执行完整的滑动窗口构建流水线。

    Args:
        split_file: 输入划分文件路径。
        output_file: 输出文件路径。
        window_size: 窗口大小。
        step_size: 滑动步长。
        config: 流水线配置。

    Returns:
        包含窗口化数据和元信息的字典。
    """
    win_config = WindowConfig(window_size=window_size, step_size=step_size)
    processor = WindowProcessor(window_config=win_config, config=config)
    return processor.process(split_file=split_file, output_file=output_file)


# =============================================================================
# 命令行入口
# =============================================================================

def main() -> None:
    """主函数：解析命令行参数并执行滑动窗口构建。"""
    parser = argparse.ArgumentParser(description="RSSI 滑动窗口构建工具")
    parser.add_argument(
        "--window-size", type=int, default=200, help="窗口大小，默认 200"
    )
    parser.add_argument(
        "--step-size", type=int, default=100, help="滑动步长，默认 100"
    )
    parser.add_argument("--input", type=Path, help="输入划分文件路径（可选）")
    parser.add_argument("--output", type=Path, help="输出文件路径（可选）")
    
    args = parser.parse_args()

    try:
        window_config = WindowConfig(
            window_size=args.window_size, step_size=args.step_size
        )
    except ValueError as e:
        print(f"配置错误: {e}")
        return

    processor = WindowProcessor(window_config=window_config)
    
    try:
        result = processor.process(split_file=args.input, output_file=args.output)
        
        meta = result["meta"]
        print("✅ 滑动窗口构建完成")
        print(f"  窗口大小: {meta['window_size']}, 步长: {meta['step_size']}")
        print(f"  训练集: {meta['num_train_windows']} 个窗口, 形状 {meta['train_feature_shape']}")
        print(f"  测试集: {meta['num_test_windows']} 个窗口, 形状 {meta['test_feature_shape']}")
        print(f"  总计: {meta['num_total_windows']} 个窗口")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()