"""RSSI 滑动窗口构建模块。

提供将时序 RSSI 数据转换为滑动窗口样本的功能，用于时间序列分析和深度学习。
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.config import PipelineConfig


@dataclass(frozen=True)
class WindowConfig:
    """滑动窗口配置。
    
    Attributes:
        window_size: 窗口大小。
        step_size: 滑动步长。
    """
    window_size: int = 200
    step_size: int = 100
    
    def __post_init__(self) -> None:
        """验证配置参数的有效性。"""
        if self.window_size < 2:
            raise ValueError(f"窗口大小必须大于 1，实际为 {self.window_size}")
        if self.step_size < 1:
            raise ValueError(f"步长必须大于 0，实际为 {self.step_size}")


class WindowBuilder:
    """滑动窗口构建器。
    
    将二维时序数据转换为滑动窗口样本。
    
    Attributes:
        config: 流水线配置对象。
        window_config: 滑动窗口配置。
    """
    
    def __init__(self, window_config: Optional[WindowConfig] = None) -> None:
        """初始化窗口构建器。
        
        Args:
            window_config: 滑动窗口配置，None 时使用默认配置。
        """
        self.config = PipelineConfig.from_root()
        self.window_config = window_config or WindowConfig()
    
    def build(self, data: np.ndarray) -> np.ndarray:
        """将二维时序数据转换为滑动窗口样本。
        
        Args:
            data: 二维数组，形状 (length, n_channels)。
            
        Returns:
            三维数组，形状 (n_windows, window_size, n_channels)。
            若数据长度小于窗口大小，返回形状为 (0, window_size, n_channels) 的空数组。
            
        Raises:
            ValueError: 当输入数据维度不为 2 时抛出。
        """
        if data.ndim != 2:
            raise ValueError(f"输入维度应为 2，实际为 {data.shape}")
        
        length, n_channels = data.shape
        window_size = self.window_config.window_size
        step_size = self.window_config.step_size
        
        if length < window_size:
            return np.empty((0, window_size, n_channels), dtype=np.float32)
        
        # 计算窗口起始索引
        starts = np.arange(0, length - window_size + 1, step_size)
        
        # 生成滑动窗口
        windows = np.stack([data[s:s + window_size] for s in starts], axis=0)
        return windows.astype(np.float32, copy=False)
    
    def build_from_samples(
        self, samples: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """从样本列表批量构建滑动窗口。
        
        Args:
            samples: 样本字典列表，每个字典需包含 "data"、"subject"、"file_name" 键。
            
        Returns:
            (features, labels, source_files) 元组：
                features: 特征数组，形状 (n_windows, window_size, n_channels)。
                labels: 标签数组，形状 (n_windows,)。
                source_files: 每个窗口的来源文件名列表。
        """
        features_list: List[np.ndarray] = []
        labels_list: List[str] = []
        files_list: List[str] = []
        
        for sample in samples:
            data = np.asarray(sample["data"], dtype=np.float32)
            windows = self.build(data)
            
            if windows.shape[0] == 0:
                continue
            
            n_windows = windows.shape[0]
            features_list.append(windows)
            labels_list.extend([str(sample["subject"])] * n_windows)
            files_list.extend([str(sample["file_name"])] * n_windows)
        
        if not features_list:
            window_size = self.window_config.window_size
            return (
                np.empty((0, window_size, 0), dtype=np.float32),
                np.empty(0, dtype=object),
                [],
            )
        
        features = np.concatenate(features_list, axis=0)
        labels = np.asarray(labels_list, dtype=object)
        
        return features, labels, files_list


class WindowProcessor:
    """滑动窗口处理流水线。
    
    整合数据加载、窗口构建和结果保存的完整流程。
    
    Attributes:
        config: 流水线配置对象。
        builder: 窗口构建器。
    """
    
    def __init__(self, window_config: Optional[WindowConfig] = None) -> None:
        """初始化窗口处理器。
        
        Args:
            window_config: 滑动窗口配置。
        """
        self.config = PipelineConfig.from_root()
        self.builder = WindowBuilder(window_config)
    
    def process(
        self,
        split_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict:
        """读取划分数据并生成滑动窗口样本。
        
        Args:
            split_file: 输入划分数据文件路径，None 时使用默认路径。
            output_file: 输出文件路径，None 时使用默认路径。
            
        Returns:
            包含窗口化数据和元信息的字典。
        """
        # 加载划分数据
        split_path = split_file or self.config.data_dir / "rssi_split.pkl"
        with split_path.open("rb") as f:
            split_data = pickle.load(f)
        
        # 构建训练集和测试集窗口
        x_train, y_train, train_files = self.builder.build_from_samples(split_data["train"])
        x_test, y_test, test_files = self.builder.build_from_samples(split_data["test"])
        
        # 构建元数据
        meta = {
            "window_size": self.builder.window_config.window_size,
            "step_size": self.builder.window_config.step_size,
            "num_train_windows": x_train.shape[0],
            "num_test_windows": x_test.shape[0],
            "num_total_windows": x_train.shape[0] + x_test.shape[0],
            "train_feature_shape": x_train.shape,
            "test_feature_shape": x_test.shape,
        }
        
        # 构建输出
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
        out_path = output_file or self.config.data_dir / "rssi_windowed.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(output, f)
        
        return output


def main() -> None:
    """主函数：解析命令行参数并执行滑动窗口构建。"""
    parser = argparse.ArgumentParser(description="RSSI 滑动窗口构建工具")
    parser.add_argument("--window-size", type=int, default=200, help="窗口大小，默认 200")
    parser.add_argument("--step-size", type=int, default=100, help="滑动步长，默认 100")
    parser.add_argument("--input", type=Path, help="输入划分文件路径（可选）")
    parser.add_argument("--output", type=Path, help="输出文件路径（可选）")
    args = parser.parse_args()
    
    # 创建配置并执行处理
    try:
        window_config = WindowConfig(
            window_size=args.window_size,
            step_size=args.step_size,
        )
    except ValueError as e:
        print(f"配置错误: {e}")
        return
    
    processor = WindowProcessor(window_config)
    result = processor.process(
        split_file=args.input,
        output_file=args.output,
    )
    
    # 输出结果摘要
    meta = result["meta"]
    print("滑动窗口构建完成")
    print(f"  窗口大小: {meta['window_size']}, 步长: {meta['step_size']}")
    print(f"  训练集: {meta['num_train_windows']} 个窗口, 形状 {meta['train_feature_shape']}")
    print(f"  测试集: {meta['num_test_windows']} 个窗口, 形状 {meta['test_feature_shape']}")
    print(f"  总计: {meta['num_total_windows']} 个窗口")


if __name__ == "__main__":
    main()