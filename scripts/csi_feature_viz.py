# -*- coding: utf-8 -*-
"""CSI 信号特征可视化 — 图A: 频域+分布, 图B: 时域动态特征。

用法:
    python scripts/csi_feature_viz.py [--subject 1] [--actions 10] [--trial 1]
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, correlate
from scipy.stats import skew, kurtosis

_FS = 100.0
_CUTOFF = 20.0
_ORDER = 4
_OUT_DIR = Path("results/figures")
_FONT = {"small": 9, "normal": 11, "large": 13, "title": 15}
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


def butter_lowpass(data, cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data.astype(np.float64), axis=0).astype(np.float32)


def _save_subplot(fig, ax_idx, exp_name, label):
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    restores = {}
    for j, other in enumerate(fig.axes):
        if j != ax_idx:
            restores[j] = other.get_visible()
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


def load_combined_csi(subject, max_actions=None, trial=1):
    disk_subj = subject + _CSI_DISK_OFFSET
    pattern = re.compile(rf"^{disk_subj}_(\d+)_(\d+)\.npy$")
    by_action = {}
    for fp in _CSI_DIR.glob(f"{disk_subj}_*.npy"):
        m = pattern.match(fp.name)
        if not m:
            continue
        by_action.setdefault(int(m.group(1)), []).append((int(m.group(2)), fp))
    if not by_action:
        raise FileNotFoundError(f"未找到用户 {subject} 的 CSI 文件 (磁盘={disk_subj})")
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


def zero_crossing_rate(signal):
    """过零率 — 信号穿过零点的频率。"""
    centered = signal - np.mean(signal)
    crossings = np.sum(np.diff(np.signbit(centered)).astype(bool))
    return crossings / len(signal)


def sliding_entropy(signal, window=200, step=50):
    """滑动窗口香农熵 — 基于直方图估计。"""
    n = len(signal)
    entropies = []
    positions = []
    for start in range(0, n - window + 1, step):
        seg = signal[start:start + window]
        counts, _ = np.histogram(seg, bins=20)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
        entropies.append(ent)
        positions.append(start + window // 2)
    return np.array(positions) / _FS, np.array(entropies)


def sliding_energy(signal, window=200, step=100):
    """滑动窗口信号能量 (均方值)。"""
    n = len(signal)
    energies = []
    positions = []
    for start in range(0, n - window + 1, step):
        seg = signal[start:start + window]
        energies.append(np.mean(seg ** 2))
        positions.append(start + window // 2)
    return np.array(positions) / _FS, np.array(energies)


# ══════════════════════════════════════════════════════════════════════
# 图 A: 频域与时域分布特征
# ══════════════════════════════════════════════════════════════════════
def figure_a(signal_1d, fs):
    """PSD + 幅度直方图 + 四项统计量。"""
    _setup_style()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

    # ── 上图: PSD ──
    ax_psd = fig.add_subplot(gs[0])
    freqs, psd = welch(signal_1d.astype(np.float64), fs=fs,
                       nperseg=min(2048, len(signal_1d)), detrend="linear")

    ax_psd.semilogy(freqs, psd, color="#1565C0", lw=1.2, label="功率谱密度")
    # 高亮低频区 (0–30 Hz, 人体运动频段)
    low_mask = freqs <= 30
    ax_psd.fill_between(freqs[low_mask], psd[low_mask],
                        alpha=0.25, color="#2E7D32",
                        label="低频分量 (≤30 Hz, 人体运动)")
    ax_psd.axvline(x=20, color="#FF6F00", linestyle="--", lw=1.8,
                   label="20 Hz (Butterworth 截止)")
    ax_psd.set_xlabel("频率 (Hz)")
    ax_psd.set_ylabel("功率谱密度 (V²/Hz)")
    ax_psd.set_title("(a) 功率谱密度 (PSD) — 低频分量主导", fontsize=_FONT["large"])
    ax_psd.set_xlim(0, min(fs / 2, 60))
    ax_psd.legend(loc="upper right", fontsize=_FONT["normal"])
    ax_psd.grid(alpha=0.2, which="both")

    # ── 下图: 幅度直方图 + 统计量 ──
    ax_hist = fig.add_subplot(gs[1])
    ax_hist.hist(signal_1d, bins=80, color="#4472C4", alpha=0.78,
                 edgecolor="white", lw=0.3, density=True)
    ax_hist.set_xlabel("信号幅度 (去均值)")
    ax_hist.set_ylabel("概率密度")
    ax_hist.set_title("(b) 信号幅度分布与统计量", fontsize=_FONT["large"])
    ax_hist.grid(axis="y", alpha=0.2)

    # 计算四项统计量
    std_val = float(np.std(signal_1d))
    skew_val = float(skew(signal_1d))
    kurt_val = float(kurtosis(signal_1d))  # excess kurtosis
    zcr_val = zero_crossing_rate(signal_1d)

    stats_text = (f"标准差 = {std_val:.4f}\n"
                  f"偏    度 = {skew_val:+.4f}\n"
                  f"峰    度 = {kurt_val:+.4f}\n"
                  f"过零率  = {zcr_val:.4f}")
    ax_hist.text(0.97, 0.95, stats_text, transform=ax_hist.transAxes,
                 fontsize=_FONT["normal"], verticalalignment="top",
                 horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="#4472C4", alpha=0.9))

    fig.suptitle("CSI 信号频域与时域分布特征", fontsize=_FONT["title"],
                 fontweight="bold")
    fig.tight_layout()

    p1 = _save_subplot(fig, 0, "CSI_FEAT", "A1_PSD")
    p2 = _save_subplot(fig, 1, "CSI_FEAT", "A2_histogram")
    return fig, p1, p2


# ══════════════════════════════════════════════════════════════════════
# 图 B: 时域动态特征
# ══════════════════════════════════════════════════════════════════════
def figure_b(signal_1d, fs):
    """原始波形 + 自相关 + 差分 + 能量条 + 熵曲线。"""
    _setup_style()
    t = np.arange(len(signal_1d)) / fs

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.2, 1.2],
                          hspace=0.45, wspace=0.3)

    # ── (a) 原始波形 (占满第一行) ──
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(t, signal_1d, color="#1565C0", lw=0.5, alpha=0.9)
    ax_raw.set_xlabel("时间 (s)")
    ax_raw.set_ylabel("幅度")
    ax_raw.set_title("(a) 原始波形 — 人体微动信号", fontsize=_FONT["large"])
    ax_raw.grid(alpha=0.15)
    ax_raw.set_xlim(t[0], t[-1])

    # ── (b) 自相关曲线 ──
    ax_acf = fig.add_subplot(gs[1, 0])
    max_lag = min(2000, len(signal_1d) // 4)
    lags = np.arange(max_lag + 1)
    acf = np.array([np.corrcoef(signal_1d[:len(signal_1d) - lag],
                                 signal_1d[lag:])[0, 1]
                    for lag in lags])
    ax_acf.plot(lags / fs, acf, color="#7B1FA2", lw=1.0)
    ax_acf.axhline(y=0, color="gray", lw=0.5, linestyle="--")
    # 标注首个过零点
    first_zero = None
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            first_zero = i
            break
    if first_zero:
        ax_acf.axvline(x=first_zero / fs, color="#FF6F00", linestyle="--",
                       lw=1.2, label=f"首个过零 ≈ {first_zero / fs:.2f}s")
        ax_acf.legend(fontsize=_FONT["small"])
    ax_acf.set_xlabel("延迟 (s)")
    ax_acf.set_ylabel("自相关系数")
    ax_acf.set_title("(b) 自相关函数 (ACF)", fontsize=_FONT["large"])
    ax_acf.grid(alpha=0.15)

    # ── (c) 差分信号 ──
    ax_diff = fig.add_subplot(gs[1, 1])
    diff_sig = np.diff(signal_1d)
    diff_std = float(np.std(diff_sig))
    ax_diff.plot(t[1:], diff_sig, color="#C62828", lw=0.4, alpha=0.85)
    ax_diff.axhline(y=0, color="gray", lw=0.5, linestyle="--")
    ax_diff.axhline(y=+diff_std, color="#FF6F00", linestyle="--", lw=1.0,
                    label=f"±1σ = ±{diff_std:.4f}")
    ax_diff.axhline(y=-diff_std, color="#FF6F00", linestyle="--", lw=1.0)
    ax_diff.set_xlabel("时间 (s)")
    ax_diff.set_ylabel("一阶差分")
    ax_diff.set_title("(c) 一阶差分 (逐点变化率)", fontsize=_FONT["large"])
    ax_diff.legend(fontsize=_FONT["small"])
    ax_diff.grid(alpha=0.15)
    ax_diff.set_xlim(t[0], t[-1])

    # ── (d) 滑动窗口能量条形图 ──
    ax_energy = fig.add_subplot(gs[2, 0])
    e_pos, e_vals = sliding_energy(signal_1d, window=int(1.0 * fs),
                                   step=int(0.5 * fs))
    norm_e = e_vals / (np.max(e_vals) + 1e-10)
    colors_e = plt.cm.YlOrRd(norm_e * 0.7 + 0.3)
    ax_energy.bar(e_pos, e_vals, width=(e_pos[1] - e_pos[0]) if len(e_pos) > 1 else 0.5,
                  color=colors_e, alpha=0.85, edgecolor="white", lw=0.2)
    ax_energy.set_xlabel("时间 (s)")
    ax_energy.set_ylabel("信号能量 (MSE)")
    ax_energy.set_title("(d) 滑动窗口能量 (窗长=1s, 步长=0.5s)",
                        fontsize=_FONT["large"])
    ax_energy.grid(axis="y", alpha=0.15)

    # ── (e) 熵值变化曲线 ──
    ax_ent = fig.add_subplot(gs[2, 1])
    ent_pos, ent_vals = sliding_entropy(signal_1d, window=int(2.0 * fs),
                                        step=int(0.25 * fs))
    ax_ent.plot(ent_pos, ent_vals, color="#00695C", lw=1.2, marker="o",
                ms=2, alpha=0.85)
    ax_ent.fill_between(ent_pos, ent_vals, alpha=0.18, color="#00695C")
    ax_ent.set_xlabel("时间 (s)")
    ax_ent.set_ylabel("香农熵 (bits)")
    ax_ent.set_title("(e) 滑动窗口熵值 (窗长=2s, 步长=0.25s)",
                     fontsize=_FONT["large"])
    ax_ent.grid(alpha=0.15)

    fig.suptitle("CSI 信号时域动态特征分析", fontsize=_FONT["title"],
                 fontweight="bold")
    fig.tight_layout()

    # 保存各子图
    paths = {}
    labels_map = {0: "B1_raw_waveform", 1: "B2_autocorr",
                  2: "B3_diff_signal", 3: "B4_energy_bars",
                  4: "B5_entropy_curve"}
    for idx, label in labels_map.items():
        paths[label] = _save_subplot(fig, idx, "CSI_FEAT", label)
    return fig, paths


def main():
    parser = argparse.ArgumentParser(description="CSI 信号特征可视化")
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
    # 低通滤波去除高频噪声 (仅用于显示更清晰的人体运动轮廓)
    filtered = butter_lowpass(signal_1d, _CUTOFF, _FS, _ORDER)
    print(f"信号范围: 原始 [{signal_1d.min():.3f}, {signal_1d.max():.3f}], "
          f"滤波后 [{filtered.min():.3f}, {filtered.max():.3f}]")

    # 图 A: PSD + 直方图
    print("\n生成图 A: 频域与时域分布特征...")
    fig_a, pa1, pa2 = figure_a(signal_1d, _FS)
    print(f"  {pa1.name}")
    print(f"  {pa2.name}")
    plt.close(fig_a)

    # 图 B: 时域动态特征 (使用滤波后信号突显人体运动轮廓)
    print("\n生成图 B: 时域动态特征...")
    fig_b, paths_b = figure_b(filtered, _FS)
    for p in paths_b.values():
        print(f"  {p.name}")
    plt.close(fig_b)

    print(f"\n全部完成, 共 7 张子图保存至 {_OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
