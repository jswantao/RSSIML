# -*- coding: utf-8 -*-
"""RSSI 数据加载模块。

本模块提供从 MAT 文件中加载和解析 RSSI 矩阵数据的功能。
支持批量处理符合命名规范的数据文件，并将数据转换为统一的 NumPy 格式。
"""
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat

# 定义公共 API
__all__ = ["DataFormatError", "load_rssi_data"]

# 编译正则表达式以提高性能，因为它是模块级常量
# 匹配格式: wipin_<subject><session>.mat
_FILE_PATTERN = re.compile(r"^wipin_([A-Za-z0-9]+)(\d+)\.mat$")


class DataFormatError(RuntimeError):
    """数据格式异常。

    当 MAT 文件格式、文件名规范或内容不符合预期时抛出。
    """
    pass


def load_rssi_data(data_dir: Path) -> List[Tuple[str, int, np.ndarray]]:
    """加载目录下所有符合规范的 RSSI 数据文件。

    扫描指定目录，解析所有符合 wipin_<subject><session>.mat 命名格式的文件，
    提取其中的 RSSI 矩阵数据。

    Args:
        data_dir: 包含 MAT 数据文件的目录路径。

    Returns:
        一个三元组列表，每个元素为 (subject, session, rssi_matrix)：
            - subject (str): 受试者标识字符串。
            - session (int): 会话编号整数。
            - rssi_matrix (np.ndarray): 二维 RSSI 数据矩阵，dtype 为 float32。

    Raises:
        FileNotFoundError: 如果目录中没有找到任何 MAT 文件。
        DataFormatError: 如果文件名格式不正确或 MAT 文件内容异常。
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    mat_files = list(data_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"目录中未找到 MAT 文件: {data_dir}")

    results: List[Tuple[str, int, np.ndarray]] = []
    
    # 建议：如果需要确定性顺序，可以使用 sorted(mat_files)
    for file_path in mat_files:
        try:
            subject, session = _parse_filename(file_path)
            matrix = _load_matrix(file_path)
            results.append((subject, session, matrix))
        except DataFormatError:
            # 重新抛出，保留原始上下文
            raise
        except Exception as e:
            # 捕获其他未知错误，包装为 DataFormatError
            raise DataFormatError(f"处理文件 {file_path} 时发生未知错误: {e}") from e

    return results


def _parse_filename(file_path: Path) -> Tuple[str, int]:
    """从文件名解析受试者标识和会话编号。

    Args:
        file_path: MAT 文件的 Path 对象。

    Returns:
        一个元组 (subject, session)。

    Raises:
        DataFormatError: 如果文件名格式不匹配预期模式。
    """
    match = _FILE_PATTERN.match(file_path.name)
    if not match:
        raise DataFormatError(
            f"文件名格式异常: {file_path.name}. "
            f"预期格式: wipin_<subject><session>.mat"
        )
    
    subject = match.group(1)
    try:
        session = int(match.group(2))
    except ValueError:
        raise DataFormatError(f"会话编号非整数: {file_path.name}")
        
    return subject, session


def _load_matrix(file_path: Path) -> np.ndarray:
    """从 MAT 文件加载 RSSI 矩阵。

    Args:
        file_path: MAT 文件的 Path 对象。

    Returns:
        二维 RSSI 数据矩阵，dtype 为 float32。

    Raises:
        DataFormatError: 如果文件加载失败、缺少 'RSSI' 键或数据维度异常。
    """
    try:
        # loadmat 返回字典，squeeze_me=True 有助于去除单维度条目
        data = loadmat(file_path, squeeze_me=True)
    except Exception as e:
        raise DataFormatError(f"MAT 文件加载失败: {file_path.name} - {e}") from e

    rssi_key = "RSSI"
    if rssi_key not in data:
        available_keys = [k for k in data if not k.startswith("__")]
        raise DataFormatError(
            f"文件 {file_path.name} 缺少 '{rssi_key}' 键。可用键: {available_keys}"
        )

    matrix = np.asarray(data[rssi_key], dtype=np.float32)
    
    # 确保是二维数组
    if matrix.ndim != 2:
        raise DataFormatError(
            f"数据维度异常: {file_path.name} 的 '{rssi_key}' 键数据形状为 {matrix.shape}，"
            f"期望二维数组 (N, M)"
        )

    return matrix