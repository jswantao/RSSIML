# -*- coding: utf-8 -*-
"""CSI 信号可视化 — 图A: 原始信号+滑动窗口, 图B: 原始vs滤波对比。

用法:
    python scripts/csi_signal_viz.py [--subject 1] [--actions 10] [--trial 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from scripts.app_utils import (
    FONT_SIZES as _FONT,
    save_experiment_subfigures,
    setup_paper_style,
)
from scripts.config import Defaults
from scripts.csi_utils import butter_lowpass, load_combined_csi

_FS = Defaults.CSI_SAMPLE_RATE
_CUTOFF = Defaults.CSI_BUTTERWORTH_CUTOFF_HZ
_ORDER = Defaults.CSI_BUTTERWORTH_ORDER
_WIN_S = 3.0
_OVERLAP = 0.5
_OUT_DIR = Path(Defaults.FIGURE_OUTPUT_DIR)


def figure_a(signal_1d: np.ndarray, fs: float, win_s: float,
             overlap: float) -> tuple:
    """图 A: 原始信号叠加滑动窗口矩形框。"""
    setup_paper_style()
    t = np.arange(len(signal_1d)) / fs
    win_samples = int(win_s * fs)
    step = int(win_samples * (1 - overlap))
    n_windows = (len(signal_1d) - win_samples) // step

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, signal_1d, color="#1565C0", lw=0.6, alpha=0.9,
            label="平均子载波幅度 (270 子载波均值)")

    stride = max(1, n_windows // 12)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, (n_windows // stride) + 1))
    for k, wi in enumerate(range(0, n_windows, stride)):
        start = wi * step
        end = start + win_samples
        color = colors[k]
        ax.axvspan(t[start], t[end], alpha=0.15, color=color, zorder=2)
        ax.annotate(f"{t[start]:.1f}s", (t[start], ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(-4, 2),
                    fontsize=_FONT["small"], ha="right", color=color, rotation=90)
        if k % 4 == 0 and k > 0:
            ax.annotate(f"{t[end]:.1f}s", (t[end], ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(-4, 2),
                        fontsize=_FONT["small"], ha="right", color=color, rotation=90)

    ax.set_xlabel("时间 (s)"); ax.set_ylabel("信号幅度")
    ax.set_title(f"(a) 原始信号与滑动窗口 (窗长={win_s}s, 重叠={overlap:.0%}, "
                 f"共 {n_windows} 个窗口)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.2)
    fig.tight_layout()
    paths = save_experiment_subfigures(fig, "CSI_VIZ", _OUT_DIR)
    return fig, paths[0] if paths else Path("")


def figure_b(signal_1d: np.ndarray, fs: float, cutoff_hz: float,
             order: int) -> tuple:
    """图 B: 原始信号 vs 滤波后信号对比 + PSD 插图。"""
    setup_paper_style()
    filtered = butter_lowpass(signal_1d, cutoff_hz=cutoff_hz, fs=fs, order=order)
    t = np.arange(len(signal_1d)) / fs

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 5)
    ax_raw_main = fig.add_subplot(gs[0, :4])
    ax_raw_psd = fig.add_subplot(gs[0, 4])
    ax_filt_main = fig.add_subplot(gs[1, :4])
    ax_filt_psd = fig.add_subplot(gs[1, 4])

    ax_raw_main.plot(t, signal_1d, color="#C62828", lw=0.5, alpha=0.85)
    ax_raw_main.set_title("(b) 原始信号 (含高频毛刺)", fontsize=_FONT["large"])
    ax_raw_main.set_ylabel("信号幅度"); ax_raw_main.grid(alpha=0.2)
    ax_raw_main.set_xlim(t[0], t[-1])

    ax_filt_main.plot(t, filtered, color="#2E7D32", lw=0.7, alpha=0.9)
    ax_filt_main.set_title(f"巴特沃斯低通滤波 (截止={cutoff_hz} Hz, {order}阶)",
                           fontsize=_FONT["large"])
    ax_filt_main.set_xlabel("时间 (s)"); ax_filt_main.set_ylabel("信号幅度")
    ax_filt_main.grid(alpha=0.2); ax_filt_main.set_xlim(t[0], t[-1])

    for ax_psd, sig, color, label in [
        (ax_raw_psd, signal_1d, "#C62828", "原始"),
        (ax_filt_psd, filtered, "#2E7D32", "滤波后"),
    ]:
        freqs, psd = welch(sig.astype(np.float64), fs=fs,
                           nperseg=min(1024, len(sig)), detrend="linear")
        ax_psd.semilogy(freqs, psd, color=color, lw=1.2, label=label)
        ax_psd.axvline(x=cutoff_hz, color="#FF6F00", linestyle="--", lw=1.5,
                       label=f"{cutoff_hz} Hz 截止")
        ax_psd.set_xlabel("频率 (Hz)", fontsize=_FONT["small"])
        ax_psd.set_ylabel("PSD", fontsize=_FONT["small"])
        ax_psd.tick_params(labelsize=_FONT["small"])
        ax_psd.legend(fontsize=_FONT["small"], loc="upper right")
        ax_psd.set_xlim(0, min(fs / 2, 60)); ax_psd.grid(alpha=0.2)

    fig.suptitle("CSI 信号: 原始 vs 巴特沃斯低通滤波 (20 Hz)",
                 fontsize=_FONT["title"], fontweight="bold")
    fig.tight_layout()
    paths = save_experiment_subfigures(fig, "CSI_VIZ", _OUT_DIR)
    return fig, paths[0] if len(paths) > 0 else Path(""), \
           paths[2] if len(paths) > 2 else Path("")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI 信号可视化")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--actions", type=int, default=10)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    print(f"加载 CSI 数据: 用户={args.subject}, 动作={args.actions}, trial={args.trial}")
    combined, labels = load_combined_csi(args.subject, args.actions, args.trial)
    print(f"拼接后 shape: {combined.shape} "
          f"({combined.shape[0] / _FS:.1f}s, {combined.shape[1]} 子载波)")

    signal_1d = combined.mean(axis=1).astype(np.float64)
    signal_1d -= signal_1d.mean()
    print(f"信号范围: [{signal_1d.min():.3f}, {signal_1d.max():.3f}]")

    print("\n生成图 A: 原始信号 + 滑动窗口...")
    fig_a, path_a = figure_a(signal_1d, _FS, _WIN_S, _OVERLAP)
    print(f"  保存: {path_a}"); plt.close(fig_a)

    print("\n生成图 B: 原始 vs 滤波对比...")
    fig_b, path_b1, path_b2 = figure_b(signal_1d, _FS, _CUTOFF, _ORDER)
    print(f"  保存: {path_b1}"); print(f"  保存: {path_b2}")
    plt.close(fig_b)
    print(f"\n全部完成, 共 3 张子图保存至 {_OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
