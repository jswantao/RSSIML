# -*- coding: utf-8 -*-
"""数据格式探查 — 读取 RSSI (MAT) 和 CSI (NPY) 数据并输出格式详情。

用法:
    python scripts/inspect_data.py
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _ROOT / "raw"
_WIFI_DIR = _ROOT / "WiFi"


def inspect_rssi(sample_file: Path) -> dict:
    """读取单个 RSSI MAT 文件并返回格式信息。"""
    mat = loadmat(sample_file)
    rssi = mat["RSSI"]
    return {
        "file": sample_file.name,
        "path": str(sample_file),
        "shape": rssi.shape,
        "dtype": str(rssi.dtype),
        "n_timesteps": rssi.shape[0],
        "n_channels": rssi.shape[1],
        "sample_rate_hz": 100.0,
        "duration_s": rssi.shape[0] / 100.0,
        "min": float(rssi.min()),
        "max": float(rssi.max()),
        "mean": float(rssi.mean()),
        "std": float(rssi.std()),
        "nonzero_ratio": float(np.count_nonzero(rssi) / rssi.size),
        "size_mb": rssi.nbytes / (1024 ** 2),
        "format": "MAT v5 (RSSI 变量)",
    }


def inspect_csi(sample_file: Path) -> dict:
    """读取单个 CSI NPY 文件并返回格式信息。"""
    arr = np.load(sample_file)  # (n_subcarriers, n_timesteps)
    return {
        "file": sample_file.name,
        "path": str(sample_file),
        "shape_raw": arr.shape,
        "shape_transposed": (arr.shape[1], arr.shape[0]),
        "dtype": str(arr.dtype),
        "n_subcarriers": arr.shape[0],
        "n_timesteps": arr.shape[1],
        "sample_rate_hz": 200.0,
        "duration_s": arr.shape[1] / 200.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nonzero_ratio": float(np.count_nonzero(arr) / arr.size),
        "size_mb": arr.nbytes / (1024 ** 2),
        "format": "NPY (subcarriers × time, 转置后为 time × subcarriers)",
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def print_dict(d: dict, indent: int = 2) -> None:
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{prefix}{k:20s}  {v:.4f}")
        else:
            print(f"{prefix}{k:20s}  {v}")


def main() -> None:
    # ── RSSI ──
    print_section("RSSI 数据格式 (MAT)")
    rssi_files = sorted(_RAW_DIR.glob("*.mat"))
    if not rssi_files:
        print(f"  未找到 MAT 文件 (预期目录: {_RAW_DIR})")
    else:
        print(f"\n  共 {len(rssi_files)} 个文件")
        for f in rssi_files[:1]:
            info = inspect_rssi(f)
            print_dict(info)
        for f in rssi_files[1:]:
            arr = loadmat(f)["RSSI"]
            print(f"\n  {f.name:30s}  shape={arr.shape!s:20s}  "
                  f"dtype={str(arr.dtype):10s}  "
                  f"{arr.nbytes / 1024**2:.1f} MB")

    # ── CSI ──
    print_section("CSI 数据格式 (NPY)")
    csi_files = sorted(_WIFI_DIR.glob("*.npy"))
    if not csi_files:
        print(f"  未找到 NPY 文件 (预期目录: {_WIFI_DIR})")
    else:
        print(f"\n  共 {len(csi_files)} 个文件")
        # 挑一个做详细检查
        sample = csi_files[len(csi_files) // 2]
        info = inspect_csi(sample)
        print_dict(info)
        # 按用户汇总
        users = {}
        for f in csi_files:
            parts = f.name.replace(".npy", "").split("_")
            if len(parts) >= 3:
                users.setdefault(parts[0], 0)
                users[parts[0]] += 1
        print(f"\n  用户分布: {len(users)} 用户, 各 {list(users.values())[0]} 文件")

    print_section("格式差异对比")
    print("""
  维度              RSSI (MAT)                    CSI (NPY)
  ────────────────  ────────────────────────────  ──────────────────────────
  原始形状          (time_steps, channels)        (subcarriers, time)
  加载后形状        直接可用                       需 .T 转置 → (time, sub)
  通道/子载波数     52 (RSSI 信道)                 270 (802.11ac 子载波)
  采样率            100 Hz                        200 Hz
  单文件时长        约 200 秒                      约 5 秒
  单文件大小        ~7.5 MB                       ~2.2 MB
  文件数量          20 (5 用户 × 4 会话)           20,900 (19 用户 × 55 动作 × 20 trials)
  数据类型          float64                       float64 → 流水线中转 float32
  加载方式          scipy.io.loadmat              np.load
  身份机构          4 会话 = 4 份独立样本           每动作 20 个 trial = 20 次重复
""")


if __name__ == "__main__":
    main()
