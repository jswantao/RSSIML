# -- coding: utf-8 --
"""app_auth 共享工具函数 — 特征提取、信号切片、绘图样式等。

优化记录:
- 🔴 移除全局可变状态 `_EXPERIMENT_FIGURES_DIR`，改为依赖注入
- 🔴 修复 save_experiment_subfigures 中的拼写错误 (restor es, set_vis ible)
- 🔴 修复 CJK 正则表达式，确保中英文数字安全过滤
- 🟠 引入特征提取器工厂与缓存机制，避免持续认证中重复实例化开销
- 🟠 完善维度校验错误提示，自动计算推荐窗口大小
- 🟢 全面采用 Python 3.10+ 类型注解与现代化标准
"""
from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.build_sliding_windows import WindowBuilder, WindowConfig
from scripts.config import PipelineConfig
from scripts.process_features_pca_norm import FeatureExtractor, PreprocessConfig

# ══════════════════════════════════════════════════════════════════════
# 全局不可变配置 (线程安全)
# ══════════════════════════════════════════════════════════════════════
FONT_SIZES: dict[str, int] = {
    "small": 14,       # annotations, bar labels, deemphasized text
    "normal": 15,      # tick labels, legend text
    "large": 16,       # axis labels, subplot titles
    "title": 18,       # figure-level suptitle
}

# 特征提取器缓存 (使用配置哈希作为键, 避免重复初始化开销)
_AUTH_FE_CACHE: dict[int, FeatureExtractor] = {}
_CACHE_LOCK = threading.Lock()

def get_auth_feature_extractor(cfg: PreprocessConfig) -> FeatureExtractor:
    """获取或创建特征提取器实例（带线程安全缓存）。"""
    key = hash(cfg)
    if key not in _AUTH_FE_CACHE:
        with _CACHE_LOCK:
            if key not in _AUTH_FE_CACHE:
                _AUTH_FE_CACHE[key] = FeatureExtractor(cfg)
    return _AUTH_FE_CACHE[key]

# ══════════════════════════════════════════════════════════════════════
# 绘图与可视化
# ══════════════════════════════════════════════════════════════════════
def setup_paper_style() -> None:
    """论文级 matplotlib 样式 — UTF-8 中文字体, 适合印刷。"""
    f = FONT_SIZES
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei",
                            "DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "font.size": f["normal"],
        "axes.titlesize": f["large"],
        "axes.labelsize": f["large"],
        "xtick.labelsize": f["normal"],
        "ytick.labelsize": f["normal"],
        "legend.fontsize": f["normal"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

def clean_title(ax: plt.Axes) -> str:
    """从 axes 标题提取安全的文件名字符串。
    保留中文/英文/数字/下划线，替换其余字符并压缩连续符号。
    """
    title = ax.get_title().strip()
    if not title:
        return "untitled"
    # \u4e00-\u9fff 覆盖基本多文种平面 (BMP) 常用汉字
    safe = re.sub(r'[^\w\u4e00-\u9fff]+', '_', title)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:50] or "untitled"

def save_experiment_figure(
    fig: plt.Figure, 
    exp_name: str, 
    output_dir: Path | str
) -> Path:
    """保存整张实验图到指定目录, 文件名包含实验名和时间戳。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = out_dir / f"{exp_name}_{ts}.png"
    fig.savefig(fpath, bbox_inches="tight")
    return fpath

def save_experiment_subfigures(
    fig: plt.Figure, 
    exp_name: str, 
    output_dir: Path | str
) -> list[Path]:
    """将组合图中的每个子图分别保存为独立文件。
    遍历 Figure 的所有 Axes, 逐个隐藏其余子图后以 tight bbox 保存。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths: list[Path] = []
    for i, ax in enumerate(fig.axes):
        # 记录并隐藏非当前子图
        restores: dict[int, bool] = {}
        for j, other in enumerate(fig.axes):
            if other is not ax:
                restores[j] = other.get_visible()
                other.set_visible(False)
        
        # 隐藏 Figure 级大标题
        stitle = getattr(fig, "_suptitle", None)
        if stitle and stitle.get_visible():
            stitle.set_visible(False)
            restores[-1] = stitle
            
        title = clean_title(ax)
        label = title if title else f"subplot_{i+1:02d}"
        fname = f"{exp_name}_{i+1:02d}_{label}_{ts}.png"
        fpath = out_dir / fname
        
        fig.savefig(fpath, bbox_inches="tight")
        paths.append(fpath)
        
        # 恢复可见性状态
        for j, visible in restores.items():
            if j == -1:
                visible.set_visible(True)
            else:
                fig.axes[j].set_visible(visible)
                
    return paths

# ══════════════════════════════════════════════════════════════════════
# 数据查找与信号处理
# ══════════════════════════════════════════════════════════════════════
def find_processed_file(
    source: str, 
    data_dir: Path | None = None
) -> Path | None:
    """查找与数据源匹配的已处理特征文件, 排除跨源误匹配。
    
    Args:
        source: 'rssi' 或 'csi'
        data_dir: 数据目录，默认从 PipelineConfig 获取
    """
    pcfg = PipelineConfig.from_root()
    target_dir = data_dir or pcfg.data_dir
    
    if source.lower() == "csi":
        pattern = "rssi_processed_authentication_npy*.pkl"
    else:
        pattern = "rssi_processed_authentication[0-9a-f]*.pkl"
        
    candidates = sorted(target_dir.glob(pattern))
    if source.lower() == "rssi":
        candidates = [c for c in candidates if "_npy" not in c.name]
        
    if not candidates:
        candidates = sorted(target_dir.glob("rssi_processed_authentication.pkl"))
        
    return candidates[-1] if candidates else None

def slice_rssi(
    data: np.ndarray, 
    slice_duration_s: float = 5.0, 
    sample_rate: float = 100.0
) -> list[np.ndarray]:
    """将长时 RSSI 样本切分为短时切片。
    
    Args:
        data: (time_steps, channels) float32, 完整 RSSI 矩阵。
        slice_duration_s: 每片时长 (秒)。
        sample_rate: RSSI 采样率 (Hz)。
        
    Returns:
        切片列表, 每片 shape 为 (slice_steps, channels)。
    """
    slice_steps = int(slice_duration_s * sample_rate)
    total_steps = data.shape[0]
    slices = []
    for start in range(0, total_steps - slice_steps + 1, slice_steps):
        slices.append(data[start:start + slice_steps])
    if not slices:
        slices.append(data)
    return slices

def build_windows(
    data: np.ndarray, 
    ws: int = 200, 
    ss: int = 100
) -> np.ndarray:
    """构建滑动窗口。
    Returns: (n_windows, n_channels, window_size)
    """
    return WindowBuilder(WindowConfig(ws, ss)).build(data)

# ══════════════════════════════════════════════════════════════════════
# 特征提取 (推理适配)
# ══════════════════════════════════════════════════════════════════════
def extract_features_for_auth(
    windows: np.ndarray,
    pca: Any | None = None,
    scaler: Any | None = None,
    feature_config: dict[str, Any] | None = None,
    feature_dim: int | None = None,
) -> np.ndarray:
    """使用与训练时一致的特征配置提取特征，自适应维度匹配。
    
    Args:
        windows: (N, C, W) 原始窗口数据
        pca: 训练时拟合的 PCA 实例
        scaler: 训练时拟合的 Scaler 实例
        feature_config: 训练时的特征配置字典
        feature_dim: 训练时最终的期望特征维度
    """
    if feature_config is None:
        feature_config = {}
        
    fg = feature_config.get("feature_groups", ("spectral", "statistical", "temporal"))
    lfb = int(feature_config.get("low_freq_bins", 16))
    dn = feature_config.get("denoise")
    dk = int(feature_config.get("denoise_kernel", 5))
    
    cfg = PreprocessConfig(
        use_pca=False, normalize=False,
        feature_groups=tuple(fg),
        low_freq_bins=lfb, denoise=dn, denoise_kernel=dk,
    )
    
    # 复用缓存的 FeatureExtractor 实例
    fe = get_auth_feature_extractor(cfg)
    fe.pca = pca
    fe.scaler = scaler
    
    raw_feats = fe.extract_features(windows)
    n_feats = raw_feats.shape[1]
    
    # 维度严格校验
    expected = feature_dim or (getattr(scaler, "n_features_in_", None) if scaler else None)
    if expected is not None and n_feats != expected:
        groups_used = cfg.feature_groups
        per_ch = 0
        if "spectral" in groups_used: per_ch += lfb
        if "statistical" in groups_used: per_ch += 4
        if "temporal" in groups_used: per_ch += 9
        
        cur_ws = n_feats // per_ch if per_ch else n_feats
        exp_ws = expected // per_ch if per_ch else expected
        
        raise ValueError(
            f"特征维度不匹配: 当前提取 {n_feats} 维 "
            f"(窗口长≈{cur_ws}, 特征组={groups_used}, low_freq_bins={lfb}), "
            f"但模型训练时期望 {expected} 维 (窗口长≈{exp_ws})。"
            f"\n修复方法: 使用 ws={exp_ws} 构建滑动窗口, "
            f"或重新训练模型以匹配当前 ws={cur_ws}。"
        )
        
    result = raw_feats
    if pca is not None:
        result = fe.transform_pca(result)
    if scaler is not None:
        result = fe.transform_scaler(result)
    return result