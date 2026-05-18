from scripts.config import Defaults
# -*- coding: utf-8 -*-
"""E1: CSI 单次认证实验 — 文件级随机划分, 同身份 vs 跨身份冒名攻击对比。"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scripts.app_utils import FONT_SIZES as FS, save_experiment_subfigures, setup_paper_style


def run_e1(self):
    """E1: CSI 单次认证 — 文件级随机划分, 在线SVM-linear, Butterworth降噪。"""
    self.logger.info("=" * 50)
    self.logger.info("E1: CSI 单次认证 (文件级划分, 10动作, 在线SVM-linear, Butterworth降噪)")
    r = self._run_svm(
        cross_activity=False,
        seed=Defaults.SEED,
        test_size=Defaults.TEST_SIZE,
        window_size=Defaults.WINDOW_SIZE,
        step_size=Defaults.STEP_SIZE,
        max_files_per_subject=100,
        threshold_method=Defaults.THRESHOLD_METHOD,
        cv_folds=Defaults.CV_FOLDS,
        use_online_svm=True,
        online_kernel="linear",
        csi_denoise="butterworth",
    )
    sm = r["system_metrics"]
    um = sm.get("user_metrics", {})

    genuine_stats = []
    impostor_stats = []
    for subj, m in um.items():
        genuine_stats.append({
            "用户": subj, "FRR": m["frr"],
            "genuine测试数": m["n_genuine_tests"],
        })
        impostor_stats.append({
            "用户": subj, "FAR": m["far"],
            "impostor测试数": m["n_impostor_tests"],
        })

    results = {
        "system": {
            "HTER": sm["mean_hter"], "FAR": sm["mean_far"],
            "FRR": sm["mean_frr"], "准确率": sm["global_accuracy"],
            "F1": sm.get("global_f1", 0),
            "HTER_std": sm.get("std_hter", 0),
            "FAR_std": sm.get("std_far", 0),
            "FRR_std": sm.get("std_frr", 0),
        },
        "genuine_per_user": sorted(genuine_stats, key=lambda x: int(x["用户"])),
        "impostor_per_user": sorted(impostor_stats, key=lambda x: int(x["用户"])),
        "per_user": um,
    }
    self.results["E1"] = results
    return results


def render_e1_results(results):
    """E1 结果: 同身份认证(FRR) vs 跨身份冒名攻击(FAR) 对比。"""
    if not isinstance(results, dict) or "system" not in results:
        return
    st.markdown("---")
    st.subheader("📊 E1: CSI 单次认证 (文件级划分, 在线SVM-linear, Butterworth降噪)")

    sys = results["system"]
    genuine = results.get("genuine_per_user", [])
    impostor = results.get("impostor_per_user", [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HTER ↓", f"{sys['HTER']:.4f}",
              help="Half Total Error Rate = (FAR+FRR)/2")
    c2.metric("FAR ↓ (冒名)", f"{sys['FAR']:.4f}",
              help="False Acceptance Rate: 冒名者被错误接受的比例")
    c3.metric("FRR ↓ (本人)", f"{sys['FRR']:.4f}",
              help="False Rejection Rate: 合法用户被错误拒绝的比例")
    c4.metric("准确率 ↑", f"{sys['准确率']:.4f}")
    c5.metric("F1 ↑", f"{sys['F1']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**同身份认证 (本人)**")
        st.caption("合法用户被错误拒绝的比例 (FRR), 越低越好")
        if genuine:
            df_g = pd.DataFrame(genuine)
            st.dataframe(df_g.style.format({"FRR": "{:.4f}"}),
                         hide_index=True, use_container_width=True)
            st.metric("平均 FRR", f"{sys['FRR']:.4f}")
    with col2:
        st.markdown("**跨身份冒名攻击 (冒名者)**")
        st.caption("攻击者伪造身份被错误接受的比例 (FAR), 越低越好")
        if impostor:
            df_i = pd.DataFrame(impostor)
            st.dataframe(df_i.style.format({"FAR": "{:.4f}"}),
                         hide_index=True, use_container_width=True)
            st.metric("平均 FAR", f"{sys['FAR']:.4f}")

    if genuine and impostor:
        setup_paper_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        users = [g["用户"] for g in genuine]
        frr_vals = [g["FRR"] for g in genuine]
        far_vals = [i["FAR"] for i in impostor]
        x = np.arange(len(users))

        w = 0.35
        ax1.bar(x - w/2, frr_vals, w, label="FRR (本人拒绝率)",
                color="#C00000", alpha=0.85, edgecolor='white')
        ax1.bar(x + w/2, far_vals, w, label="FAR (冒名接受率)",
                color="#4472C4", alpha=0.85, edgecolor='white')
        ax1.axhline(y=sys["FRR"], color="#C00000", linestyle='--', lw=1, alpha=0.5)
        ax1.axhline(y=sys["FAR"], color="#4472C4", linestyle='--', lw=1, alpha=0.5)
        ax1.set_xticks(x); ax1.set_xticklabels(users, fontsize=FS["small"])
        ax1.set_ylabel("错误率", fontsize=FS["large"])
        
        ax1.legend(fontsize=FS["normal"])
        ax1.grid(axis='y', alpha=0.2)

        for i, u in enumerate(users):
            ax2.scatter(frr_vals[i], far_vals[i], s=60, color='#C00000',
                        zorder=3, edgecolors='white', linewidth=0.5)
            ax2.annotate(u, (frr_vals[i], far_vals[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=FS["small"])
        ax2.axhline(y=sys["FAR"], color="#4472C4", linestyle='--', lw=1,
                    label=f'平均 FAR={sys["FAR"]:.4f}')
        ax2.axvline(x=sys["FRR"], color="#C00000", linestyle='--', lw=1,
                    label=f'平均 FRR={sys["FRR"]:.4f}')
        ax2.set_xlabel("FRR (本人拒绝率)", fontsize=FS["large"])
        ax2.set_ylabel("FAR (冒名接受率)", fontsize=FS["large"])
        
        ax2.legend(fontsize=FS["small"])
        ax2.grid(alpha=0.2)
        ax2.fill_between([0, min(0.1, max(frr_vals)*1.2)],
                         0, min(0.1, max(far_vals)*1.2),
                         alpha=0.05, color='#2E7D32')
        ax2.text(min(0.05, max(frr_vals)*0.6), min(0.05, max(far_vals)*0.6),
                 '理想区域', fontsize=FS["small"], color='#2E7D32', alpha=0.5)

        fig.suptitle("E1: CSI 同身份认证 vs 跨身份冒名攻击 (文件级划分)",
                     fontsize=FS["title"], fontweight='bold')
        fig.tight_layout(); save_experiment_subfigures(fig, "E1"); st.pyplot(fig); plt.close(fig)

    far_val = sys["FAR"]
    frr_val = sys["FRR"]
    if far_val < 0.05 and frr_val < 0.05:
        grade, color = "优秀", "#2E7D32"
    elif far_val < 0.1 and frr_val < 0.1:
        grade, color = "良好", "#1565C0"
    elif far_val < 0.2 and frr_val < 0.2:
        grade, color = "一般", "#ED7D31"
    else:
        grade, color = "较差", "#C00000"
    st.markdown(f"""
    <div style="padding:15px; border-radius:8px; background:{color}18;
                border-left:4px solid {color}; margin:10px 0;">
        <strong style="color:{color};">冒名攻击防御评级: {grade}</strong><br>
        <span style="font-size:0.9rem; color:#555;">
        冒名者被错误接受的概率为 <b>{far_val:.2%}</b>,
        合法用户被错误拒绝的概率为 <b>{frr_val:.2%}</b>。
        </span>
    </div>
    """, unsafe_allow_html=True)
