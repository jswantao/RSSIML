# -*- coding: utf-8 -*-
"""E3: 注册时长影响实验 — 不同训练样本量下的认证性能学习曲线。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scripts.app_utils import FONT_SIZES as FS, save_experiment_subfigures, setup_paper_style


def run_e3(self):
    """E3: 注册时长影响 — 不同训练样本量 (仅 SVM)。"""
    self.logger.info("=" * 50)
    self.logger.info("E3: 注册时长影响实验 (SVM)")
    results = []
    sizes = [20, 40, 80, 100, 140, 200]

    for mf in sizes:
        self._check_cancelled()
        self.logger.info(f"E3: 每用户{mf}样本...")
        r = self._run_svm(max_files_per_subject=mf)
        sm = r["system_metrics"]
        results.append({
            "每用户文件数": mf,
            "HTER": sm["mean_hter"], "FAR": sm["mean_far"],
            "FRR": sm["mean_frr"], "准确率": sm["global_accuracy"],
            "HTER_std": sm.get("std_hter", 0),
            "FAR_std": sm.get("std_far", 0),
            "FRR_std": sm.get("std_frr", 0),
        })

    self.results["E3"] = results
    return results


def render_e3_results(results):
    """E3 结果: 注册时长学习曲线 (SVM)。"""
    if not isinstance(results, list) or len(results) < 2:
        return
    st.markdown("---")
    st.subheader("📊 E3: 注册时长影响 (SVM)")

    df = pd.DataFrame(results)
    st.dataframe(df.style.format(
        {"HTER": "{:.4f}", "FAR": "{:.4f}", "FRR": "{:.4f}", "准确率": "{:.4f}"}
    ), hide_index=True, use_container_width=True)

    setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df["每用户文件数"].values.astype(float)

    for metric, color, marker in [("HTER", "#C00000", "o"),
                                   ("FAR", "#4472C4", "s"),
                                   ("FRR", "#ED7D31", "^")]:
        y = df[metric].values.astype(float)
        ax1.plot(x, y, marker=marker, color=color, lw=2, ms=8, label=metric)
        std_col = f"{metric}_std"
        if std_col in df.columns:
            std_v = df[std_col].values.astype(float)
            if not np.all(std_v == 0):
                ax1.fill_between(x, np.maximum(0, y - std_v), y + std_v,
                                 color=color, alpha=0.12)
    ax1.set_xlabel("训练样本 / 用户", fontsize=FS["large"])
    ax1.set_ylabel("错误率", fontsize=FS["large"])
    
    ax1.legend(fontsize=FS["normal"]); ax1.grid(alpha=0.2)
    ax1.set_ylim(0, min(1.0, df[["HTER", "FAR", "FRR"]].max().max() * 1.35))

    acc = df["准确率"].values.astype(float)
    ax2.plot(x, acc, 'D-', color='#2E7D32', lw=2, ms=8)
    ax2.set_xlabel("训练样本 / 用户", fontsize=FS["large"])
    ax2.set_ylabel("准确率", fontsize=FS["large"])
    
    ax2.grid(alpha=0.2)
    ax2.set_ylim(max(0.0, acc.min() - 0.05), min(1.0, acc.max() + 0.05))

    fig.suptitle("E3: 注册时长对认证性能的影响 (SVM)", fontsize=FS["title"], fontweight='bold')
    fig.tight_layout(); save_experiment_subfigures(fig, "E3"); st.pyplot(fig); plt.close(fig)
