# -*- coding: utf-8 -*-
"""E5: RSSI 切片式单次认证实验 — 每切片独立认证, 汇总评估整体性能。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scripts.app_utils import FONT_SIZES as FS, save_experiment_subfigures, setup_paper_style
from scripts.models import svm_scores
from scripts.config import Defaults


def run_e5(self):
    """E5: RSSI 切片式单次认证 — 固定时长切片, 每片独立判定, 汇总评估。

    与推理阶段 `_render_single_auth` 切片模式完全一致的逻辑:
      1. 训练 SVM (匹配基础训练 RSSI 默认配置)
      2. 加载每用户原始 .mat → 按固定时长切分不重叠切片
      3. 每切片: build_windows → extract_features → mean_score ≥ threshold
      4. 汇总: 切片通过率、每用户统计、整体指标

    对比至少三组切片时长。
    """
    self.logger.info("=" * 50)
    self.logger.info("E5: RSSI 切片式单次认证 (匹配推理切片模式)")

    # ── 阶段1: 模型训练 (与推理一致) ─────────────────────────────────
    self.logger.info("训练 SVM 模型 (匹配基础训练 RSSI 默认配置)...")
    r = self._run_svm(
        seed=Defaults.SEED, test_size=Defaults.TEST_SIZE, window_size=Defaults.WINDOW_SIZE, step_size=Defaults.STEP_SIZE,
        threshold_method=Defaults.THRESHOLD_METHOD, cv_folds=Defaults.CV_FOLDS,
        use_pca=False, use_online_svm=False,
        feature_groups=("spectral", "statistical", "temporal"),
        cross_activity=False, clean_intermediate=True,
    )
    sm = r["system_metrics"]
    verifiers = r.get("verifiers", {})
    if not verifiers:
        self.results["E5"] = {}
        return {}

    # ── 阶段2: 多组切片时长对比 ─────────────────────────────────────
    slice_durations = [3, 5, 10, 20]  # 至少三组
    all_results = {}

    for dur in slice_durations:
        self._check_cancelled()
        self.logger.info(f"切片时长={dur}s 单次认证评估...")
        per_user = {}
        all_decisions = []  # 收集所有切片判定 (用于计算整体 FAR/FRR)

        for subj in sorted(verifiers.keys(), key=int):
            self._check_cancelled()
            ctx = self._prepare_auth_context(r, subj=subj)
            if ctx is None:
                self.logger.warning(f"  用户 {subj} 数据加载失败, 跳过")
                continue

            _, verifier, threshold, pca, scaler, fc, fd, raw = ctx

            # 与推理切片模式完全一致的逻辑
            from scripts.app_utils import slice_rssi, build_windows, extract_features_for_auth

            slices = slice_rssi(raw, slice_duration_s=float(dur))
            if len(slices) < 2:
                continue

            slice_decisions = []
            window_accepts = []
            for seg in slices:
                windows = build_windows(seg)
                if windows.shape[0] == 0:
                    continue
                feats = extract_features_for_auth(
                    windows, pca, scaler, fc, feature_dim=fd)
                scores = svm_scores(verifier, feats)
                mean_s = float(np.mean(scores))
                is_ok = mean_s >= threshold
                slice_decisions.append(is_ok)
                # 切片内窗口级接受率
                window_accepts.append(float(np.mean(scores >= threshold)))

            if not slice_decisions:
                continue

            n_accept = sum(slice_decisions)
            n_total = len(slice_decisions)
            per_user[subj] = {
                "用户": subj,
                "切片总数": n_total,
                "接受切片数": n_accept,
                "拒绝切片数": n_total - n_accept,
                "切片通过率": n_accept / n_total,
                "平均窗口接受率": float(np.mean(window_accepts)) if window_accepts else 0,
                "阈值": float(threshold),
            }
            all_decisions.extend(slice_decisions)

        # 整体 FAR/FRR 近似: 同身份用户切片判定为接受率
        overall_accept = float(np.mean(all_decisions)) if all_decisions else 0
        all_results[f"{dur}s"] = {
            "切片时长(s)": dur,
            "HTER": sm["mean_hter"],
            "FAR": sm["mean_far"],
            "FRR": sm["mean_frr"],
            "准确率": sm["global_accuracy"],
            "总切片数": len(all_decisions),
            "通过切片数": int(np.sum(all_decisions)),
            "拒绝切片数": len(all_decisions) - int(np.sum(all_decisions)),
            "整体切片通过率": overall_accept,
            "每用户": per_user,
        }
        self.logger.info(f"  {dur}s: {len(per_user)} 用户, "
                        f"通过率={overall_accept:.1%}")

    self.results["E5"] = all_results
    return all_results


def render_e5_results(results):
    """E5 结果: 切片式单次认证 — 切片时长对比 + 每用户分布。"""
    if not results or not isinstance(results, dict):
        return
    st.markdown("---")
    st.subheader("📊 E5: RSSI 切片式单次认证 (推理切片模式)")
    st.caption("与推理阶段 `_render_single_auth` 切片模式完全一致的判定逻辑。")

    dur_keys = sorted(results.keys(),
                      key=lambda k: results[k]["切片时长(s)"])

    # ── 汇总表格 ─────────────────────────────────────────────────────
    tbl_rows = []
    for k in dur_keys:
        v = results[k]
        tbl_rows.append({
            "切片时长": k,
            "总切片数": v["总切片数"],
            "通过切片": v["通过切片数"],
            "拒绝切片": v["拒绝切片数"],
            "整体切片通过率": f"{v['整体切片通过率']:.1%}",
            "HTER": v["HTER"], "FAR": v["FAR"], "FRR": v["FRR"],
        })
    st.dataframe(pd.DataFrame(tbl_rows).style.format(
        {"HTER": "{:.4f}", "FAR": "{:.4f}", "FRR": "{:.4f}"}
    ), hide_index=True, use_container_width=True)

    if len(dur_keys) < 1:
        return

    # ── 切片通过率 vs 切片时长 ──────────────────────────────────────
    setup_paper_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    dur_vals = [results[k]["切片时长(s)"] for k in dur_keys]
    pass_rates = [results[k]["整体切片通过率"] for k in dur_keys]
    total_slices = [results[k]["总切片数"] for k in dur_keys]

    # (a) 切片通过率 vs 切片时长
    ax1.plot(dur_vals, pass_rates, 'o-', color='#2E7D32', lw=2.5, ms=10)
    ax1.set_xlabel("切片时长 (s)", fontsize=FS["large"])
    ax1.set_ylabel("整体切片通过率", fontsize=FS["large"])
    
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.2)
    for i, (d, r) in enumerate(zip(dur_vals, pass_rates)):
        ax1.annotate(f"{r:.1%}", (d, r), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=FS["normal"], color='#2E7D32')

    # (b) 切片总数 vs 切片时长
    ax2.bar(dur_vals, total_slices, color='#4472C4', alpha=0.85,
            edgecolor='white')
    for i, (d, c) in enumerate(zip(dur_vals, total_slices)):
        ax2.text(d, c + 0.5, str(c), ha='center', fontsize=FS["normal"], color='#4472C4')
    ax2.set_xlabel("切片时长 (s)", fontsize=FS["large"])
    ax2.set_ylabel("总切片数", fontsize=FS["large"])
    
    ax2.grid(axis='y', alpha=0.2)

    # (c) 每用户切片通过率分布 (首个时长)
    first_dur = dur_keys[0]
    per_user = results[first_dur].get("每用户", {})
    if per_user:
        users = sorted(per_user.keys(), key=int)
        user_rates = [per_user[u]["切片通过率"] for u in users]
        colors_u = ['#2E7D32' if r >= 0.5 else '#C62828' for r in user_rates]
        ax3.bar(range(len(users)), user_rates, color=colors_u, alpha=0.85,
                edgecolor='white')
        ax3.axhline(y=0.5, color='#FF6F00', linestyle='--', lw=1.5,
                    label='决策边界 (0.5)')
        ax3.set_xticks(range(len(users)))
        ax3.set_xticklabels(users, fontsize=FS["normal"])
        ax3.set_xlabel("用户", fontsize=FS["large"])
        ax3.set_ylabel("切片通过率", fontsize=FS["large"])
        
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=FS["small"])
        ax3.grid(axis='y', alpha=0.2)

    # (d) 所有时长的每用户热力图
    if per_user:
        users = sorted(per_user.keys(), key=int)
        heatmap = np.zeros((len(users), len(dur_keys)))
        for j, k in enumerate(dur_keys):
            pu = results[k].get("每用户", {})
            for i, u in enumerate(users):
                heatmap[i, j] = pu.get(u, {}).get("切片通过率", 0) if pu else 0

        im = ax4.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax4.set_xticks(range(len(dur_keys)))
        ax4.set_xticklabels(dur_keys, fontsize=FS["normal"])
        ax4.set_yticks(range(len(users)))
        ax4.set_yticklabels(users, fontsize=FS["normal"])
        ax4.set_xlabel("切片时长", fontsize=FS["large"])
        ax4.set_ylabel("用户", fontsize=FS["large"])
        
        plt.colorbar(im, ax=ax4, label="切片通过率")

    fig.suptitle("E5: RSSI 切片式单次认证 — 切片时长对比",
                 fontsize=FS["title"], fontweight='bold')
    fig.tight_layout()
    save_experiment_subfigures(fig, "E5")
    st.pyplot(fig)
    plt.close(fig)

    # ── 汇总指标 ─────────────────────────────────────────────────────
    best_dur = max(dur_keys, key=lambda k: results[k]["整体切片通过率"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最佳切片时长",
              f"{results[best_dur]['切片时长(s)']}s",
              f"通过率 {results[best_dur]['整体切片通过率']:.1%}")
    c2.metric("HTER (训练)",
              f"{results[dur_keys[0]]['HTER']:.4f}")
    c3.metric("总切片数 (所有时长)",
              f"{sum(results[k]['总切片数'] for k in dur_keys)}")
    c4.metric("评估用户数",
              f"{len(per_user)}")

    st.caption(
        "切片时长越大 → 每片包含更多信号 → 单次判定更可靠 (通过率高)。"
        "切片时长越小 → 判定更密集 → 可用于更精细的时序分析。"
        "与推理阶段切片模式一致: 每切片独立 build_windows → 特征提取 → "
        "mean_score ≥ threshold → 接受/拒绝。")
