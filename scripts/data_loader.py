"""RSSI 数据处理模块。

本模块提供从 MAT 文件中加载和解析 RSSI 矩阵数据的功能。
支持批量处理符合命名规范的数据文件。
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat


class DataFormatError(RuntimeError):
    """数据格式异常。

    当 MAT 文件格式或内容不符合预期时抛出。
    """


# 文件名模式：wipin_<subject><session>.mat
_FILE_PATTERN = re.compile(r"^wipin_([A-Za-z0-9]+)(\d+)\.mat$")
_RSSI_KEY = "RSSI"


def load_rssi_data(data_dir: Path) -> List[Tuple[str, int, np.ndarray]]:
    """加载目录下所有符合规范的 RSSI 数据文件。

    扫描指定目录，解析所有符合 wipin_<subject><session>.mat 命名格式的文件，
    提取其中的 RSSI 矩阵数据。

    Args:
        data_dir: 包含 MAT 数据文件的目录路径。

    Returns:
        三元组列表，每个元素为 (subject, session, rssi_matrix)：
            subject: 受试者标识字符串。
            session: 会话编号整数。
            rssi_matrix: 二维 RSSI 数据矩阵，dtype 为 float32。

    Raises:
        FileNotFoundError: 目录中无 MAT 文件时抛出。
        DataFormatError: 文件名格式不符或数据内容异常时抛出。

    Examples:
        >>> data = load_rssi_data(Path("./data"))
        >>> for subject, session, matrix in data:
        ...     print(f"{subject}_session{session}: {matrix.shape}")
    """
    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"目录中未找到 MAT 文件: {data_dir}")

    results = []
    for file_path in mat_files:
        subject, session = _parse_filename(file_path)
        matrix = _load_matrix(file_path)
        results.append((subject, session, matrix))

    return results


def _parse_filename(file_path: Path) -> Tuple[str, int]:
    """从文件名解析受试者标识和会话编号。

    Args:
        file_path: MAT 文件路径。

    Returns:
        (subject, session) 元组。

    Raises:
        DataFormatError: 文件名格式不匹配预期模式时抛出。
    """
    match = _FILE_PATTERN.match(file_path.name)
    if not match:
        raise DataFormatError(f"文件名格式异常: {file_path.name}")
    return match.group(1), int(match.group(2))


def _load_matrix(file_path: Path) -> np.ndarray:
    """从 MAT 文件加载 RSSI 矩阵。

    Args:
        file_path: MAT 文件路径。

    Returns:
        二维 RSSI 数据矩阵，dtype 为 float32。

    Raises:
        DataFormatError: 文件加载失败、缺少 RSSI 键或数据维度异常时抛出。
    """
    try:
        data = loadmat(file_path)
    except Exception as e:
        raise DataFormatError(f"MAT 文件加载失败: {file_path.name} - {e}") from e

    if _RSSI_KEY not in data:
        available = [k for k in data if not k.startswith("__")]
        raise DataFormatError(
            f"缺少 '{_RSSI_KEY}' 键: {file_path.name}，可用键: {available}"
        )

    matrix = np.asarray(data[_RSSI_KEY], dtype=np.float32)
    if matrix.ndim != 2:
        raise DataFormatError(
            f"数据维度异常: {file_path.name} 形状 {matrix.shape}，期望二维"
        )

    return matrix