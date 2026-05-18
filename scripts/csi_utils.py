# -*- coding: utf-8 -*-
"""CSI 数据工具 — 加载、滤波、信号处理 (消除跨脚本重复)。"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from scripts.config import Defaults, PipelineConfig


def load_combined_csi(subject: int, max_actions: int | None = None,
                      trial: int = 1) -> tuple[np.ndarray, list[str]]:
    """加载指定用户的 CSI 数据, 每个动作取一个 trial 沿时间轴拼接。

    CSI NPY 文件 shape 为 (n_subcarriers, n_timesteps), 加载后转置为
    (n_timesteps, n_subcarriers)。

    Returns:
        (time_steps, n_subcarriers) 拼接数组, 动作标签列表。
    """
    csi_dir = PipelineConfig.from_root().npy_dir
    disk_subj = subject + Defaults.CSI_DISK_OFFSET
    pattern = re.compile(rf"^{disk_subj}_(\d+)_(\d+)\.npy$")

    by_action: dict[int, list[Path]] = {}
    for fp in csi_dir.glob(f"{disk_subj}_*.npy"):
        m = pattern.match(fp.name)
        if not m:
            continue
        by_action.setdefault(int(m.group(1)), []).append((int(m.group(2)), fp))
    if not by_action:
        raise FileNotFoundError(
            f"未找到用户 {subject} (磁盘编号={disk_subj}) 的 CSI 文件, "
            f"请确认 {csi_dir}/ 目录存在 {disk_subj}_*.npy 文件")

    selected = []
    labels = []
    for action in sorted(by_action):
        files = sorted(by_action[action], key=lambda x: x[0])
        idx = min(trial - 1, len(files) - 1)
        if idx >= 0:
            selected.append(files[idx][1])
            labels.append(f"动作{action}")
        if max_actions and len(selected) >= max_actions:
            break

    arrays = [np.load(fp).T for fp in selected]
    return np.concatenate(arrays, axis=0).astype(np.float32), labels


def butter_lowpass(data: np.ndarray, cutoff_hz: float | None = None,
                   fs: float | None = None, order: int | None = None,
                   normalized_cutoff: float | None = None) -> np.ndarray:
    """巴特沃斯低通滤波 — 支持物理频率 (Hz) 和归一化截止频率两种调用方式。

    Args:
        data: (n_timesteps, ...) 输入信号。
        cutoff_hz: 物理截止频率 (Hz), 需同时提供 fs。
        fs: 采样率 (Hz)。
        order: 滤波器阶数, 默认 4。
        normalized_cutoff: 归一化截止频率 (0~1, 相对 Nyquist)。优先级高于 cutoff_hz+fs。

    Returns:
        滤波后同形状数组 (float32)。
    """
    order = order or Defaults.CSI_BUTTERWORTH_ORDER
    if normalized_cutoff is not None:
        nc = normalized_cutoff
    elif cutoff_hz is not None and fs is not None:
        nc = cutoff_hz / (0.5 * fs)
    else:
        nc = Defaults.CSI_BUTTERWORTH_CUTOFF_HZ / (0.5 * (fs or Defaults.CSI_SAMPLE_RATE))
    if nc >= 1.0:
        return data.astype(np.float32)
    sos = butter(order, nc, btype="low", output="sos")
    return sosfiltfilt(sos, data.astype(np.float64), axis=0).astype(np.float32)


def zero_crossing_rate(signal: np.ndarray) -> float:
    """过零率 — 信号穿过零点的频率。"""
    centered = signal - np.mean(signal)
    crossings = np.sum(np.diff(np.signbit(centered)).astype(bool))
    return float(crossings / len(signal))


def sliding_entropy(signal: np.ndarray, window: int = 200, step: int = 50,
                    fs: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口香农熵 — 基于直方图估计。

    Returns:
        (positions_s, entropy_bits) 位置 (秒) 和熵值数组。
    """
    n = len(signal)
    entropies, positions = [], []
    for start in range(0, n - window + 1, step):
        seg = signal[start:start + window]
        counts, _ = np.histogram(seg, bins=20)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))
        positions.append(start + window // 2)
    return np.array(positions) / fs, np.array(entropies)


def sliding_energy(signal: np.ndarray, window: int = 200, step: int = 100,
                   fs: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口信号能量 (均方值)。

    Returns:
        (positions_s, energy) 位置 (秒) 和能量数组。
    """
    n = len(signal)
    energies, positions = [], []
    for start in range(0, n - window + 1, step):
        seg = signal[start:start + window]
        energies.append(float(np.mean(seg ** 2)))
        positions.append(start + window // 2)
    return np.array(positions) / fs, np.array(energies)
