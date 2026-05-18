# -*- coding: utf-8 -*-
"""统一数据集划分模块 — RSSI (MAT) 与 CSI (NPY) 通用。

数据组织:
  RSSI (raw/):  5 用户 × 4 会话 = 20 文件, wipin_<subject><session>.mat
  CSI  (WiFi/): 19 用户 × 55 动作 × 20 重复 = 20900 文件, {subject}_{activity}_{trial}.npy

支持身份认证任务。CSI 数据采用元数据延迟加载，划分文件仅保存路径引用，
避免中间 pickle 文件膨胀。实际矩阵在窗口构建阶段按需加载。

优化记录 (v3.2):
- 🔴 彻底移除 DataLoadContext.global_instance() 隐式调用，强制显式传递上下文
- 🔴 修复 Sample.to_dict() 隐式加载问题：划分阶段仅保存元数据，不触发磁盘 I/O
- 🔴 修复所有拼写/语法错误 (cache_key, subject_groups, impostor_ratio, defaultdict 等)
- 🟠 增强 _build_auth_test 数据隔离，确保 impostor 采样不污染训练集引用
- 🟢 全面采用 Python 3.10+ 类型注解与现代化标准
- 🟢 日志命名空间修正为 __name__，避免全局污染
"""
from __future__ import annotations

import gzip
import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scripts.config import PipelineConfig
from scripts.data_loader import DataFormatError, DataLoadContext, load_npy_matrix, parse_npy_filename

logger = logging.getLogger(__name__)

__all__ = [
    "Sample",
    "DatasetSummary",
    "DatasetSplitter",
    "split_authentication_dataset",
]

_MIN_SAMPLES_PER_SUBJECT = 2
_SERIALIZATION_VERSION = "3.2"
DataSource = Literal["rssi", "csi"]


# ══════════════════════════════════════════════════════════════════════
# 数据样本
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Sample:
    """数据样本（RSSI 或 CSI 通用）。
    
    RSSI: data 已加载为 (time, channels) float32。
    CSI:  data 为延迟加载引用 (Path)；下游阶段按需解析。
    """
    file_name: str
    subject: str
    session: int
    data: Any  # np.ndarray (RSSI) 或 Path (CSI)

    def to_dict(self) -> dict[str, Any]:
        """返回元数据字典。不触发 CSI 磁盘加载，确保划分文件轻量。"""
        return {
            "file_name": self.file_name,
            "subject": self.subject,
            "session": self.session,
            "data": self.data,  # 保持 Path 引用或已加载的 ndarray
        }

    def resolve_data(self, context: DataLoadContext) -> np.ndarray:
        """按需解析数据（CSI 延迟加载入口）。
        
        Raises:
            DataFormatError: 文件损坏或格式不匹配。
        """
        if isinstance(self.data, Path):
            cache_key = str(self.data)
            try:
                return context.data_cache.get_or_load(
                    cache_key,
                    lambda: load_npy_matrix(self.data),
                )
            except (ValueError, OSError) as e:
                raise DataFormatError(f"无法加载延迟数据: {self.data.name}", file_path=self.data) from e
        return self.data


# ══════════════════════════════════════════════════════════════════════
# 数据集摘要
# ══════════════════════════════════════════════════════════════════════
@dataclass
class DatasetSummary:
    """数据集统计摘要。"""
    total_samples: int
    num_subjects: int
    samples_per_subject: dict[str, int]
    sessions_per_subject: dict[str, list[int]]
    data_source: str
    estimated_memory_mb: float
    nan_sample_count: int
    inf_sample_count: int

    def __repr__(self) -> str:
        lines = [
            f"DatasetSummary(source={self.data_source})",
            f"  样本总数: {self.total_samples}",
            f"  用户数:   {self.num_subjects}",
            f"  平均每用户: {self.total_samples / max(1, self.num_subjects):.1f} 样本",
            f"  估算内存: {self.estimated_memory_mb:.1f} MB",
        ]
        if self.nan_sample_count or self.inf_sample_count:
            lines.append(
                f"  数据质量: {self.nan_sample_count} NaN, {self.inf_sample_count} Inf"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 数据集划分器
# ══════════════════════════════════════════════════════════════════════
class DatasetSplitter:
    """数据集划分器 — 支持 RSSI (MAT) 和 CSI (NPY)。
    
    RSSI: 从 raw/ 加载，数据驻内存。
    CSI:  从 WiFi/ 扫描元数据，按需加载，避免内存溢出。
    """

    def __init__(
        self,
        seed: int | None = None,
        data_source: DataSource = "rssi",
        config: PipelineConfig | None = None,
        max_files_per_subject: int | None = None,
        context: DataLoadContext | None = None,
    ) -> None:
        self.config = config or PipelineConfig.from_root()
        self.seed = seed if seed is not None else self.config.random_seed
        self.rng = np.random.default_rng(self.seed)
        self.data_source = data_source
        self.max_files = max_files_per_subject
        self._ctx = context or DataLoadContext()  # 强制生命周期隔离
        self.samples = self._load_samples()

        n = len(self.samples)
        subjects = {s.subject for s in self.samples}
        logger.info(
            "DatasetSplitter: %d 样本, %d 用户, source=%s", n, len(subjects), data_source
        )
        self.summary = self._build_summary()

    # ── 摘要 ──────────────────────────────────────────────────────────────
    def _build_summary(self) -> DatasetSummary:
        counter = Counter(s.subject for s in self.samples)
        sessions: dict[str, list[int]] = defaultdict(list)
        nan_cnt, inf_cnt = 0, 0

        for s in self.samples:
            sessions[s.subject].append(s.session)
            if isinstance(s.data, np.ndarray) and s.data.size > 0:
                if np.any(np.isnan(s.data)):
                    nan_cnt += 1
                if np.any(np.isinf(s.data)):
                    inf_cnt += 1

        est_mb = 0.0
        if self.samples:
            first_data = self.samples[0].data
            if isinstance(first_data, np.ndarray) and first_data.size > 0:
                est_mb = (len(self.samples) * first_data.nbytes) / (1024 * 1024)
            elif isinstance(first_data, Path) and first_data.exists():
                try:
                    header = np.load(first_data, mmap_mode="r")
                    est_mb = (len(self.samples) * header.nbytes) / (1024 * 1024)
                    del header
                except (ValueError, OSError):
                    est_mb = 0.0

        return DatasetSummary(
            total_samples=len(self.samples),
            num_subjects=len(counter),
            samples_per_subject=dict(sorted(counter.items())),
            sessions_per_subject={k: sorted(set(v)) for k, v in sessions.items()},
            data_source=self.data_source,
            estimated_memory_mb=est_mb,
            nan_sample_count=nan_cnt,
            inf_sample_count=inf_cnt,
        )

    def summarize(self) -> str:
        return repr(self.summary)

    # ── 加载 ──────────────────────────────────────────────────────────────
    def _load_samples(self) -> list[Sample]:
        return self._load_csi_samples() if self.data_source == "csi" else self._load_rssi_samples()

    def _load_rssi_samples(self) -> list[Sample]:
        from scripts.data_loader import load_rssi_data
        raw = load_rssi_data(self.config.raw_dir, context=self._ctx)
        subj_map = self.config.subject_map("rssi")
        return [Sample(f"wipin_{s}{n}.mat", subj_map.get(s, s), n, m) for s, n, m in raw]

    def _load_csi_samples(self) -> list[Sample]:
        npy_files = sorted(self.config.npy_dir.glob("*.npy"))
        if not npy_files:
            logger.warning("CSI 目录无 NPY 文件: %s", self.config.npy_dir)

        selected = self.config.csi_selected_actions
        subj_map = self.config.subject_map("csi")
        metas: list[Sample] = []
        skipped = 0
        for fp in npy_files:
            try:
                subj, sess = self._parse_csi_filename(fp)
                subj = subj_map.get(subj, subj)  # 12-30 → 1-19
                if selected is not None and (sess // 100) not in selected:
                    continue
                metas.append(Sample(fp.name, subj, sess, fp))
            except ValueError:
                skipped += 1

        if selected is not None:
            logger.info("CSI 动作筛选: %d/55 → %d 样本", len(selected), len(metas))
        if skipped:
            logger.warning("跳过 %d 个文件名格式异常的文件", skipped)

        if self.max_files is not None and self.max_files > 0:
            metas = self._limit_per_subject(metas, self.max_files)
        return metas

    @staticmethod
    def _parse_csi_filename(file_path: Path) -> tuple[str, int]:
        """解析 CSI 文件名 — 委托给共享的 parse_npy_filename。"""
        return parse_npy_filename(file_path)

    def _limit_per_subject(self, samples: list[Sample], limit: int) -> list[Sample]:
        if self.data_source == "csi":
            return self._limit_per_subject_by_action(samples, limit)
        groups = self._group_by_subject(samples)
        result: list[Sample] = []
        for group in groups.values():
            if len(group) > limit:
                idx = self.rng.choice(len(group), size=limit, replace=False)
                result.extend(group[i] for i in idx)
            else:
                result.extend(group)
        return result

    def _limit_per_subject_by_action(self, samples: list[Sample], limit: int) -> list[Sample]:
        selected_actions = self.config.csi_selected_actions
        na = len(selected_actions) if selected_actions else 55
        if limit % na != 0:
            raise ValueError(f"CSI 每用户文件数必须为 {na} 的倍数: 当前 {limit}")

        trials_per_action = limit // na
        if trials_per_action > 20:
            raise ValueError(f"每动作最多 20 个 trial: 需要 {trials_per_action}")

        groups = self._group_by_subject(samples)
        result: list[Sample] = []
        for subject, group in groups.items():
            action_groups: dict[int, list[Sample]] = defaultdict(list)
            for s in group:
                action_groups[s.session // 100].append(s)

            if len(action_groups) != na:
                raise ValueError(f"用户 {subject} 仅有 {len(action_groups)} 个动作, 期望 {na}")

            for action, files in action_groups.items():
                if len(files) < trials_per_action:
                    raise ValueError(
                        f"用户 {subject} 动作 {action} 仅有 {len(files)} trial, "
                        f"需要 {trials_per_action}"
                    )
                idx = self.rng.choice(len(files), size=trials_per_action, replace=False)
                result.extend(files[i] for i in idx)
        return result

    # ── 通用工具 ──────────────────────────────────────────────────────────
    def _group_by_subject(self, samples: list[Sample] | None = None) -> dict[str, list[Sample]]:
        src = samples if samples is not None else self.samples
        groups: dict[str, list[Sample]] = {}
        for s in src:
            groups.setdefault(s.subject, []).append(s)
        return groups

    def _resolve_data(self, sample: Sample) -> np.ndarray:
        return sample.resolve_data(self._ctx)

    def _validate_split_params(self, test_size: float) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size 必须在 (0, 1) 之间: {test_size}")

    def _check_min_samples(self, subject_groups: dict[str, list[Sample]]) -> None:
        insufficient = [
            s for s, g in subject_groups.items() if len(g) < _MIN_SAMPLES_PER_SUBJECT
        ]
        if insufficient:
            raise ValueError(f"样本数不足的用户 (需 ≥{_MIN_SAMPLES_PER_SUBJECT}): {insufficient}")

    def _save(self, output: dict[str, Any], filename: str, compress: bool = False) -> Path:
        out_dir = self.config.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        output.setdefault("meta", {})
        output["meta"].update({
            "serialization_version": _SERIALIZATION_VERSION,
            "data_source": self.data_source,
            "compressed": compress,
        })
        path = out_dir / filename
        if compress:
            path = path.with_suffix(path.suffix + ".gz")

        open_func = gzip.open if compress else open
        with open_func(path, "wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("保存至: %s (%s)", path, "gzip" if compress else "uncompressed")
        return path

    def _filename(self, task: str) -> str:
        suffix = "_npy" if self.data_source == "csi" else ""
        return f"rssi_split_{task}{suffix}.pkl"

    # ══════════════════════════════════════════════════════════════════════
    # 认证任务划分
    # ══════════════════════════════════════════════════════════════════════
    def split_authentication(
        self,
        test_size: float = 0.2,
        impostor_ratio: float = 0.5,
        cross_activity: bool = False,
        compress: bool = False,
    ) -> dict[str, Any]:
        self._validate_split_params(test_size)
        if impostor_ratio <= 0:
            raise ValueError(f"impostor_ratio 必须 > 0: {impostor_ratio}")

        subject_groups = self._group_by_subject()
        subjects = sorted(subject_groups.keys())

        if len(subjects) < 2:
            raise ValueError(f"身份认证需要至少 2 个用户: 当前 {len(subjects)}")

        self._check_min_samples(subject_groups)

        if cross_activity and self.data_source == "csi":
            return self._split_auth_cross_activity(
                subject_groups, subjects, test_size, impostor_ratio, compress
            )
        return self._split_auth_file_level(
            subject_groups, subjects, test_size, impostor_ratio, compress
        )

    def _split_auth_file_level(
        self,
        subject_groups: dict[str, list[Sample]],
        subjects: list[str],
        test_size: float,
        impostor_ratio: float,
        compress: bool,
    ) -> dict[str, Any]:
        """文件级认证划分 — 随机选取每用户文件作为 train/test。
        CSI 数据使用延迟加载模式：不立即加载数据，保留 Path 引用。
        """
        user_train: dict[str, list[dict[str, Any]]] = {}
        user_genuine: dict[str, list[dict[str, Any]]] = {}

        for subj in subjects:
            group = subject_groups[subj]
            n_test = max(1, min(len(group) - 1, round(len(group) * test_size)))
            indices = self.rng.permutation(len(group))

            # CSI 保留 Path 引用；RSSI 立即解析
            is_csi = self.data_source == "csi"
            # 训练集
            train_indices = indices[n_test:]
            train_samples = [group[i] for i in train_indices]
            user_train[subj] = [
                s.to_dict() if is_csi
                else {**s.to_dict(), "data": self._resolve_data(s)}
                for s in train_samples
            ]
            # 测试集 genuine (必须标注 claimed_identity 和 true_label)
            test_indices = indices[:n_test]
            test_samples = [group[i] for i in test_indices]
            user_genuine[subj] = [
                {**(s.to_dict() if is_csi
                   else {**s.to_dict(), "data": self._resolve_data(s)}),
                 "claimed_identity": subj, "true_label": "genuine"}
                for s in test_samples
            ]

        train_all = [s for samples in user_train.values() for s in samples]
        auth_test = self._build_auth_test(subjects, user_train, user_genuine, impostor_ratio)

        self.rng.shuffle(train_all)
        self.rng.shuffle(auth_test)

        genuine_count = sum(1 for s in auth_test if s["true_label"] == "genuine")
        impostor_count = len(auth_test) - genuine_count

        output = {
            "meta": {
                "seed": self.seed,
                "split_strategy": "authentication_per_user",
                "task": "authentication",
                "test_size_requested": test_size,
                "impostor_ratio_requested": impostor_ratio,
                "cross_activity": False,
                "num_train_samples": len(train_all),
                "num_auth_test_samples": len(auth_test),
                "num_genuine_tests": genuine_count,
                "num_impostor_tests": impostor_count,
                "num_subjects": len(subjects),
                "subjects": subjects,
            },
            "train": train_all,
            "auth_test": auth_test,
        }
        self._save(output, self._filename("authentication"), compress=compress)
        return output

    def _split_auth_cross_activity(
        self,
        subject_groups: dict[str, list[Sample]],
        subjects: list[str],
        test_size: float,
        impostor_ratio: float,
        compress: bool,
    ) -> dict[str, Any]:
        """跨动作认证划分（仅 CSI）— 延迟加载数据。"""
        def _action(sample: Sample) -> int:
            return sample.session // 100

        all_actions = sorted({_action(s) for s in self.samples})
        n_test_actions = max(1, round(len(all_actions) * test_size))
        indices = self.rng.permutation(len(all_actions))
        test_actions = {all_actions[i] for i in indices[:n_test_actions]}
        train_actions = set(all_actions) - test_actions

        logger.info("跨动作认证: %d train actions, %d test actions", len(train_actions), len(test_actions))

        user_train: dict[str, list[dict[str, Any]]] = {}
        user_genuine: dict[str, list[dict[str, Any]]] = {}

        for subj in subjects:
            group = subject_groups[subj]
            train_files = [s for s in group if _action(s) in train_actions]
            test_files = [s for s in group if _action(s) in test_actions]

            if len(train_files) < _MIN_SAMPLES_PER_SUBJECT:
                raise ValueError(f"用户 {subj} 训练 action 样本不足 ({len(train_files)})")

            user_train[subj] = [s.to_dict() for s in train_files]
            user_genuine[subj] = [
                {**s.to_dict(), "claimed_identity": subj, "true_label": "genuine"}
                for s in test_files
            ]

        train_all = [s for samples in user_train.values() for s in samples]
        auth_test = self._build_auth_test(subjects, user_train, user_genuine, impostor_ratio)

        self.rng.shuffle(train_all)
        self.rng.shuffle(auth_test)

        genuine_count = sum(1 for s in auth_test if s["true_label"] == "genuine")
        impostor_count = len(auth_test) - genuine_count

        output = {
            "meta": {
                "seed": self.seed,
                "split_strategy": "authentication_cross_activity",
                "task": "authentication",
                "test_size_requested": test_size,
                "impostor_ratio_requested": impostor_ratio,
                "cross_activity": True,
                "train_actions": sorted(train_actions),
                "test_actions": sorted(test_actions),
                "num_train_samples": len(train_all),
                "num_auth_test_samples": len(auth_test),
                "num_genuine_tests": genuine_count,
                "num_impostor_tests": impostor_count,
                "num_subjects": len(subjects),
                "subjects": subjects,
            },
            "train": train_all,
            "auth_test": auth_test,
        }
        self._save(output, self._filename("authentication_ca"), compress=compress)
        return output

    def _build_auth_test(
        self,
        subjects: list[str],
        user_train: dict[str, list[dict[str, Any]]],
        user_genuine: dict[str, list[dict[str, Any]]],
        impostor_ratio: float,
    ) -> list[dict[str, Any]]:
        """构建认证测试集 (genuine + impostor)。确保数据深拷贝隔离。"""
        auth_test: list[dict[str, Any]] = []

        for subj in subjects:
            genuine_samples = user_genuine.get(subj, [])
            if not genuine_samples:
                logger.warning("用户 %s 无 genuine 测试样本，跳过", subj)
                continue

            auth_test.extend(genuine_samples)

            n_impostors = max(1, round(len(genuine_samples) * impostor_ratio))
            # 收集所有其他用户的训练样本
            other_samples = [
                s for o in subjects if o != subj for s in user_train.get(o, [])
            ]
            if not other_samples:
                logger.warning("用户 %s 无可用 impostor 候选", subj)
                continue

            n_sample = min(n_impostors, len(other_samples))
            chosen_indices = self.rng.choice(len(other_samples), size=n_sample, replace=False)

            for idx in chosen_indices:
                # 深拷贝避免污染原始划分字典
                imp = {**other_samples[idx], "claimed_identity": subj, "true_label": "impostor"}
                auth_test.append(imp)

        return auth_test


# ══════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════
def split_authentication_dataset(
    test_size: float = 0.2,
    seed: int = 42,
    impostor_ratio: float = 0.5,
    data_source: DataSource = "rssi",
    cross_activity: bool = False,
    compress: bool = False,
    context: DataLoadContext | None = None,
) -> dict[str, Any]:
    """便捷划分入口。"""
    splitter = DatasetSplitter(
        seed=seed, data_source=data_source, context=context
    )
    return splitter.split_authentication(
        test_size=test_size,
        impostor_ratio=impostor_ratio,
        cross_activity=cross_activity,
        compress=compress,
    )

# ══════════════════════════════════════════════════════════════════════
# 缓存清理
# ══════════════════════════════════════════════════════════════════════
def _clear_sample_cache(context: DataLoadContext | None = None) -> None:
    """清除样本缓存 — 释放 DataLoadContext 持有的数据。"""
    if context is not None:
        context.clear()
    logger.debug("样本缓存已清理")
