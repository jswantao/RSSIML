# -*- coding: utf-8 -*-
"""E2: RSSI 持续认证实验 — 完整复现基础训练→推理阶段的全流程。"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scripts.app_utils import FONT_SIZES as FS, save_experiment_subfigures, setup_paper_style


def run_e2(self):
    """E2: 全流程复现 — 训练(匹配基础训练)→持续认证(匹配推理阶段)。

    1. 使用与基础训练标签页完全一致的参数训练 SVM 模型
    2. 对每位注册用户加载其原始 .mat 文件
    3. 以推理阶段 `_render_continuous_auth` 完全一致的逻辑执行持续认证
    4. 输出每用户汇总 + 代表用户分数曲线图
    """
    self.logger.info("=" * 50)
    self.logger.info("E2: RSSI 持续认证 (基础训练→推理 全流程复现)")

    # ── 阶段1: 模型训练 (与基础训练标签页完全一致) ─────────────────
    self.logger.info("训练 SVM 模型 (匹配基础训练 RSSI 默认配置)...")
    r = self._run_svm(
        seed=42, test_size=0.2, window_size=200, step_size=100,
        threshold_method="youden", cv_folds=5,
        use_pca=False, use_online_svm=False,
        feature_groups=("spectral", "statistical", "temporal"),
        cross_activity=False, clean_intermediate=True,
    )
    sm = r["system_metrics"]
    verifiers = r.get("verifiers", {})
    if not verifiers:
        self.results["E2"] = {}
        return {}

    # ── 阶段2: 持续认证 (逐用户, 与推理阶段完全一致) ──────────────
    self.logger.info(f"持续认证: {len(verifiers)} 用户, 平滑窗口=10 (推理默认)...")

    per_user = {}
    for subj in sorted(verifiers.keys(), key=int):
        self._check_cancelled()
        self.logger.info(f"  用户 {subj} 持续认证...")
        ctx = self._prepare_auth_context(r, subj=subj)
        if ctx is None:
            self.logger.warning(f"  用户 {subj} 数据加载失败, 跳过")
            continue

        _, verifier, threshold, pca, scaler, fc, fd, raw = ctx

        # 与推理阶段 _render_continuous_auth 完全一致的逻辑
        from scripts.app_utils import build_windows, extract_features_for_auth
        from scripts.models import svm_scores

        windows = build_windows(raw)
        n = windows.shape[0]
        if n < 2:
            per_user[subj] = {"用户": subj, "总窗口数": n, "错误": "窗口不足"}
            continue

        feats = extract_features_for_auth(
            windows, pca, scaler, fc, feature_dim=fd)
        all_scores = svm_scores(verifier, feats)

        # 滑动平均 (与推理阶段 ws=10 完全一致)
        smooth_ws = 10
        smoothed = np.array([
            np.mean(all_scores[max(0, i - smooth_ws + 1):i + 1])
            for i in range(n)
        ])
        decisions = smoothed >= threshold

        # 统计 (与推理阶段完全一致)
        longest = streak = 0
        for d in decisions:
            if d:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        switches = sum(1 for i in range(1, n)
                       if decisions[i] != decisions[i - 1])

        per_user[subj] = {
            "用户": subj,
            "总窗口数": n,
            "接受率": float(np.mean(decisions)),
            "最终决策": "接受" if decisions[-1] else "拒绝",
            "最长连续接受": longest,
            "状态切换次数": switches,
            "阈值": float(threshold),
            "HTER_train": sm.get("user_metrics", {}).get(subj, {}).get("hter", None),
            "原始分数": all_scores.tolist() if n <= 2000 else all_scores[:2000].tolist(),
            "平滑分数": smoothed.tolist() if n <= 2000 else smoothed[:2000].tolist(),
            "决策序列": [int(d) for d in decisions],
        }

    self.results["E2"] = {
        "system": sm,
        "per_user": per_user,
    }
    self.logger.info(f"E2 完成: {len(per_user)} 用户持续认证")
    return self.results["E2"]


def render_e2_results(results):
    """E2 结果: 全流程复现 — 每用户汇总 + 代表用户分数曲线(推理风格)。"""
    if not results or not isinstance(results, dict):
        return
    st.markdown("---")
    st.subheader("📊 E2: RSSI 持续认证 (基础训练→推理 全流程复现)")

    per_user = results.get("per_user", {})
    system = results.get("system", {})
    if not per_user:
        st.info("无持续认证结果。")
        return

    # ── 系统指标 ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HTER", f"{system.get('mean_hter', 0):.4f}")
    c2.metric("FAR", f"{system.get('mean_far', 0):.4f}")
    c3.metric("FRR", f"{system.get('mean_frr', 0):.4f}")
    c4.metric("准确率", f"{system.get('global_accuracy', 0):.4f}")

    # ── 每用户汇总表 ─────────────────────────────────────────────────
    st.markdown("### 每用户持续认证结果")
    st.caption("与推理阶段 `_render_continuous_auth` 完全一致的逻辑。")

    rows = []
    for subj in sorted(per_user.keys(), key=int):
        v = per_user[subj]
        rows.append({
            "用户": v["用户"],
            "总窗口数": v.get("总窗口数", "—"),
            "接受率": f"{v.get('接受率', 0):.1%}"
                      if v.get("接受率") is not None else "—",
            "最长连续接受": v.get("最长连续接受"),
            "状态切换": v.get("状态切换次数"),
            "最终决策": v.get("最终决策"),
            "阈值": f"{v.get('阈值', 0):.4f}",
            "HTER(训练)": f"{v.get('HTER_train', 0):.4f}"
                          if v.get("HTER_train") is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── 代表用户分数曲线 (推理风格: 分数 + 决策条) ──────────────────
    users = sorted(per_user.keys(), key=int)
    if not users:
        return

    # 选最优和最差用户展示
    best_user = max(users, key=lambda u: per_user[u].get("接受率", 0))
    worst_user = min(users, key=lambda u: per_user[u].get("接受率", 0))
    display_users = [best_user]
    if worst_user != best_user:
        display_users.append(worst_user)

    st.markdown("### 代表用户分数曲线 (推理阶段风格)")
    st.caption("上图: 原始分数(灰) + 平滑分数(蓝) + 阈值(橙); 下图: 绿色=接受, 红色=拒绝")

    setup_paper_style()
    n_users = len(display_users)
    fig, axes = plt.subplots(n_users * 2, 1, figsize=(14, 4.5 * n_users),
                             squeeze=False)

    for row, subj in enumerate(display_users):
        v = per_user[subj]
        raw_scores = np.array(v.get("原始分数", []))
        smooth_scores = np.array(v.get("平滑分数", []))
        decisions_arr = v.get("决策序列", [])
        threshold = v.get("阈值", 0.5)
        n = len(raw_scores)

        if n == 0:
            continue

        ax_score = axes[row * 2][0]
        ax_bar = axes[row * 2 + 1][0]
        t = np.arange(n)

        # 分数曲线
        ax_score.fill_between(t, threshold, 1.0, alpha=0.08, color='#2E7D32')
        ax_score.fill_between(t, 0, threshold, alpha=0.08, color='#C62828')
        ax_score.plot(t, raw_scores, alpha=0.25, color='#9E9E9E', lw=0.5,
                      label='原始分数 (逐窗口)')
        ax_score.plot(t, smooth_scores, color='#1565C0', lw=1.8,
                      label=f'平滑分数 (窗口=10)')
        ax_score.axhline(y=threshold, color='#FF6F00', linestyle='--', lw=1.8,
                         label=f'阈值 = {threshold:.4f}')
        ax_score.set_ylabel("认证分数", fontsize=FS["large"])
        hter_u = v.get("HTER_train", 0) or 0
        ax_score.set_title(
            f"用户 {subj} | 接受率={v.get('接受率', 0):.1%} | "
            f"训练HTER={hter_u:.4f} | 最终={v.get('最终决策')}",
            fontsize=FS["large"])
        ax_score.legend(loc='upper right', fontsize=FS["normal"], framealpha=0.9)
        ax_score.set_ylim(0, 1.05)
        ax_score.grid(True, alpha=0.25)

        # 决策条
        colors = ['#2E7D32' if d else '#C62828' for d in decisions_arr]
        ax_bar.bar(t, np.ones(n), width=1.0, color=colors, alpha=0.7)
        ax_bar.set_xlabel("窗口序号", fontsize=FS["large"])
        ax_bar.set_ylabel("决策", fontsize=FS["large"])
        ax_bar.set_yticks([])
        ax_bar.set_ylim(0, 1)

    fig.suptitle("E2: RSSI 持续认证 — 基础训练→推理 全流程复现",
                 fontsize=FS["title"], fontweight='bold')
    fig.tight_layout()
    save_experiment_subfigures(fig, "E2")
    st.pyplot(fig)
    plt.close(fig)

    # ── 汇总 ─────────────────────────────────────────────────────────
    accept_rates = [v.get("接受率", 0) for v in per_user.values()
                    if v.get("接受率") is not None]
    avg_accept = np.mean(accept_rates) if accept_rates else 0
    n_accept = sum(1 for v in per_user.values() if v.get("最终决策") == "接受")
    n_total = len(per_user)

    c1, c2, c3 = st.columns(3)
    c1.metric("平均接受率", f"{avg_accept:.1%}")
    c2.metric("最终决策=接受", f"{n_accept}/{n_total} 用户")
    c3.metric("HTER (训练)", f"{system.get('mean_hter', 0):.4f}")

    st.caption(
        "持续认证模拟采用与推理阶段 `_render_continuous_auth` 完全一致的方案: "
        "构建滑动窗口 (ws=200, ss=100) → 特征提取 → SVM 逐窗口评分 → "
        "前向滑动平均 (窗口=10) → 平滑分数 ≥ 阈值 → 逐窗口接受/拒绝判定。")
