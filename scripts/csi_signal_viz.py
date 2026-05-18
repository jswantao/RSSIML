# -*- coding: utf-8 -*-
"""CSI 信号可视化 — 图A: 原始信号+滑动窗口, 图B: 原始vs滤波对比。

用法:
    python scripts/csi_signal_viz.py [--subject 1] [--actions 10] [--trial 1]
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

_FS = 100.0        # CSI 采样率 (Hz)
_CUTOFF = 20.0     # Butterworth 截止频率 (Hz)
_ORDER = 4          # 滤波器阶数
_WIN_S = 3.0        # 滑动窗口时长 (s)
_OVERLAP = 0.5      # 窗口重叠比例
_OUT_DIR = Path("results/figures")
_FONT = {"small": 9, "normal": 11, "large": 13, "title": 15}

# CSI 用户映射: UI 编号 → 磁盘编号 (1→12, ..., 19→30)
_CSI_DISK_OFFSET = 11
_CSI_DIR = Path("WiFi")


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": _FONT["normal"],
        "axes.titlesize": _FONT["large"],
        "axes.labelsize": _FONT["large"],
        "xtick.labelsize": _FONT["normal"],
        "ytick.labelsize": _FONT["normal"],
        "legend.fontsize": _FONT["normal"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def butter_lowpass(data: np.ndarray, cutoff_hz: float, fs: float,
                   order: int = 4) -> np.ndarray:
    """巴特沃斯低通滤波 — 使用 sosfiltfilt 实现零相位。"""
    nyq = 0.5 * fs
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data.astype(np.float64), axis=0).astype(np.float32)


def load_combined_csi(subject: int, max_actions: int | None = None,
                      trial: int = 1) -> tuple[np.ndarray, list[str]]:
    """加载指定用户的 CSI 数据, 每个动作取一个 trial 沿时间轴拼接。

    CSI NPY 文件 shape 为 (n_subcarriers, n_timesteps), 加载后转置为
    (n_timesteps, n_subcarriers)。

    Returns:
        (time_steps, n_subcarriers) 拼接数组, 动作标签列表。
    """
    disk_subj = subject + _CSI_DISK_OFFSET  # 1→12, 2→13, ..., 19→30
    pattern = re.compile(rf"^{disk_subj}_(\d+)_(\d+)\.npy$")

    by_action: dict[int, list[Path]] = {}
    for fp in _CSI_DIR.glob(f"{disk_subj}_*.npy"):
        m = pattern.match(fp.name)
        if not m:
            continue
        by_action.setdefault(int(m.group(1)), []).append((int(m.group(2)), fp))
    if not by_action:
        raise FileNotFoundError(f"未找到用户 {subject} (磁盘编号={disk_subj}) 的 CSI 文件, "
                              f"请确认 {_CSI_DIR}/ 目录存在 {disk_subj}_*.npy 文件")

    selected_files = []
    labels = []
    for action in sorted(by_action):
        files = sorted(by_action[action], key=lambda x: x[0])
        idx = min(trial - 1, len(files) - 1)
        if idx >= 0:
            selected_files.append(files[idx][1])
            labels.append(f"动作{action}")
        if max_actions and len(selected_files) >= max_actions:
            break

    arrays = []
    for fp in selected_files:
        arr = np.load(fp)  # (n_subcarriers, n_timesteps)
        arr = arr.T         # → (n_timesteps, n_subcarriers)
        arrays.append(arr)
    combined = np.concatenate(arrays, axis=0).astype(np.float32)
    return combined, labels


def _save_subplot(fig: plt.Figure, ax_idx: int, exp_name: str, label: str):
    """保存单个子图为独立文件。"""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for j, other in enumerate(fig.axes):
        if j != ax_idx:
            other.set_visible(False)
    stitle = getattr(fig, "_suptitle", None)
    st_visible = None
    if stitle and stitle.get_visible():
        st_visible = True
        stitle.set_visible(False)
    fname = f"{exp_name}_{label}_{ts}.png"
    fpath = _OUT_DIR / fname
    fig.savefig(fpath, bbox_inches="tight")
    for j, other in enumerate(fig.axes):
        other.set_visible(True)
    if st_visible:
        stitle.set_visible(True)
    return fpath


def figure_a(signal_1d: np.ndarray, fs: float, win_s: float,
             overlap: float):
    """图 A: 原始信号叠加滑动窗口矩形框。"""
    _setup_style()
    t = np.arange(len(signal_1d)) / fs

    win_samples = int(win_s * fs)
    step = int(win_samples * (1 - overlap))
    n_windows = (len(signal_1d) - win_samples) // step

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, signal_1d, color="#1565C0", lw=0.6, alpha=0.9,
            label="平均子载波幅度 (270 子载波均值)")

    # 最多显示 12 个窗口以避免视觉拥挤
    stride = max(1, n_windows // 12)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, (n_windows // stride) + 1))
    for k, wi in enumerate(range(0, n_windows, stride)):
        start = wi * step
        end = start + win_samples
        color = colors[k]
        ax.axvspan(t[start], t[end], alpha=0.15, color=color, zorder=2)
        ax.annotate(f"{t[start]:.1f}s", (t[start], ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(-4, 2), fontsize=_FONT["small"],
                    ha="right", color=color, rotation=90)
        if k == (n_windows // stride) // stride:
            ax.annotate(f"{t[end]:.1f}s", (t[end], ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(-4, 2),
                        fontsize=_FONT["small"], ha="right", color=color, rotation=90)

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("信号幅度")
    ax.set_title(f"(a) 原始信号与滑动窗口 (窗长={win_s}s, 重叠={overlap:.0%}, "
                 f"共 {n_windows} 个窗口)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, _save_subplot(fig, 0, "CSI_VIZ", "A_windows")


def figure_b(signal_1d: np.ndarray, fs: float, cutoff_hz: float,
             order: int):
    """图 B: 原始信号 vs 滤波后信号对比 + PSD 插图。"""
    _setup_style()
    filtered = butter_lowpass(signal_1d, cutoff_hz, fs, order)
    t = np.arange(len(signal_1d)) / fs

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 5)

    # 上图: 原始信号 (主图 + PSD 插图)
    ax_raw_main = fig.add_subplot(gs[0, :4])
    ax_raw_psd = fig.add_subplot(gs[0, 4])

    # 下图: 滤波后信号
    ax_filt_main = fig.add_subplot(gs[1, :4])
    ax_filt_psd = fig.add_subplot(gs[1, 4])

    # ── 原始信号 ──
    ax_raw_main.plot(t, signal_1d, color="#C62828", lw=0.5, alpha=0.85)
    ax_raw_main.set_title("(b) 原始信号 (含高频毛刺)", fontsize=_FONT["large"])
    ax_raw_main.set_ylabel("信号幅度")
    ax_raw_main.grid(alpha=0.2)
    ax_raw_main.set_xlim(t[0], t[-1])

    # ── 滤波后信号 ──
    ax_filt_main.plot(t, filtered, color="#2E7D32", lw=0.7, alpha=0.9)
    ax_filt_main.set_title(f"巴特沃斯低通滤波 (截止={cutoff_hz} Hz, {order}阶)",
                           fontsize=_FONT["large"])
    ax_filt_main.set_xlabel("时间 (s)")
    ax_filt_main.set_ylabel("信号幅度")
    ax_filt_main.grid(alpha=0.2)
    ax_filt_main.set_xlim(t[0], t[-1])

    # ── PSD 插图 ──
    for ax_psd, sig, color, label in [
        (ax_raw_psd, signal_1d, "#C62828", "原始"),
        (ax_filt_psd, filtered, "#2E7D32", "滤波后"),
    ]:
        freqs, psd = welch(sig.astype(np.float64), fs=fs, nperseg=min(1024, len(sig)),
                           detrend="linear")
        ax_psd.semilogy(freqs, psd, color=color, lw=1.2, label=label)
        ax_psd.axvline(x=cutoff_hz, color="#FF6F00", linestyle="--", lw=1.5,
                       label=f"{cutoff_hz} Hz 截止")
        ax_psd.set_xlabel("频率 (Hz)", fontsize=_FONT["small"])
        ax_psd.set_ylabel("PSD", fontsize=_FONT["small"])
        ax_psd.tick_params(labelsize=_FONT["small"])
        ax_psd.legend(fontsize=_FONT["small"], loc="upper right")
        ax_psd.set_xlim(0, min(fs / 2, 60))
        ax_psd.grid(alpha=0.2)

    fig.suptitle("CSI 信号: 原始 vs 巴特沃斯低通滤波 (20 Hz)",
                 fontsize=_FONT["title"], fontweight="bold")
    fig.tight_layout()

    a_path = _save_subplot(fig, 0, "CSI_VIZ", "B1_raw")
    b_path = _save_subplot(fig, 2, "CSI_VIZ", "B2_filtered")
    return fig, a_path, b_path


def main():
    parser = argparse.ArgumentParser(description="CSI 信号可视化")
    parser.add_argument("--subject", type=int, default=1, help="用户编号 (1-19)")
    parser.add_argument("--actions", type=int, default=10, help="拼接动作数")
    parser.add_argument("--trial", type=int, default=1, help="每动作 trial 编号")
    args = parser.parse_args()

    print(f"加载 CSI 数据: 用户={args.subject}, 动作数={args.actions}, trial={args.trial}")
    combined, labels = load_combined_csi(args.subject, args.actions, args.trial)
    print(f"拼接后 shape: {combined.shape} "
          f"({combined.shape[0] / _FS:.1f}s, {combined.shape[1]} 子载波)")

    # 平均所有子载波得到 1D 代表信号
    signal_1d = combined.mean(axis=1).astype(np.float64)
    # 去均值 (移除 DC 偏移)
    signal_1d -= signal_1d.mean()
    print(f"信号范围: [{signal_1d.min():.3f}, {signal_1d.max():.3f}]")

    # 图 A
    print("\n生成图 A: 原始信号 + 滑动窗口...")
    fig_a, path_a = figure_a(signal_1d, _FS, _WIN_S, _OVERLAP)
    print(f"  保存: {path_a}")
    plt.close(fig_a)

    # 图 B
    print("\n生成图 B: 原始 vs 滤波对比...")
    fig_b, path_b1, path_b2 = figure_b(signal_1d, _FS, _CUTOFF, _ORDER)
    print(f"  保存: {path_b1}")
    print(f"  保存: {path_b2}")
    plt.close(fig_b)

    print(f"\n全部完成, 共 3 张子图保存至 {_OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
