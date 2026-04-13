# -*- coding: utf-8 -*-
"""RSSI 数据集划分模块。

提供分类任务和开集识别任务的数据集划分功能，支持按人员分层的文件粒度划分。
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from scripts.config import PipelineConfig
from scripts.data_loader import load_rssi_data

# 定义公共 API
__all__ = [
    "Sample",
    "DatasetSplitter",
    "split_classification_dataset",
    "split_identification_dataset",
]


@dataclass(frozen=True)
class Sample:
    """RSSI 数据样本。

    Attributes:
        file_name: 原始文件名。
        subject: 受试者标识符。
        session: 会话编号。
        data: RSSI 矩阵数据。
    """
    file_name: str
    subject: str
    session: int
    data: np.ndarray


class DatasetSplitter:
    """数据集划分器。

    提供分类任务和开集识别任务的数据集划分功能。

    Attributes:
        config: 流水线配置对象。
        samples: 所有样本列表。
        rng: NumPy 随机数生成器。
        seed: 随机种子。
    """

    def __init__(self, seed: int = 42, config: Optional[PipelineConfig] = None) -> None:
        """初始化数据集划分器。

        Args:
            seed: 随机种子，用于结果复现。
            config: 流水线配置对象。如果为 None，则使用默认配置。
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.config = config if config is not None else PipelineConfig.from_root()
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Sample]:
        """加载所有 RSSI 数据样本。

        Returns:
            Sample 对象列表。
        """
        raw_data = load_rssi_data(self.config.raw_dir)
        return [
            Sample(
                file_name=f"wipin_{subject}{session}.mat",
                subject=subject,
                session=session,
                data=matrix,
            )
            for subject, session, matrix in raw_data
        ]

    def _save_split(self, output: dict, filename: str) -> None:
        """保存划分结果到文件。

        Args:
            output: 包含划分结果的字典。
            filename: 输出文件名。
        """
        out_path = self.config.data_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(output, f)

    def _group_by_subject(self) -> Dict[str, List[Sample]]:
        """按受试者分组样本。

        Returns:
            受试者到其 Sample 对象列表的映射字典。
        """
        groups: Dict[str, List[Sample]] = {}
        for sample in self.samples:
            groups.setdefault(sample.subject, []).append(sample)
        return groups

    @staticmethod
    def _create_sample_payload(sample: Sample) -> dict:
        """将 Sample 对象转换为字典格式。

        Args:
            sample: Sample 对象。

        Returns:
            包含样本信息的字典。
        """
        return {
            "file_name": sample.file_name,
            "subject": sample.subject,
            "session": sample.session,
            "data": sample.data,
        }

    @staticmethod
    def _count_subject_distribution(samples: List[dict]) -> Dict[str, int]:
        """统计样本中受试者的分布。

        Args:
            samples: 样本字典列表。

        Returns:
            受试者到样本数量的映射字典，按键排序。
        """
        distribution: Dict[str, int] = {}
        for sample in samples:
            subject = str(sample["subject"])
            distribution[subject] = distribution.get(subject, 0) + 1
        return dict(sorted(distribution.items()))

    def split_classification(self, test_size: float = 0.2) -> dict:
        """分类任务：按人员分层的文件粒度划分。

        为身份分类任务创建数据划分，确保每个人员的文件在训练集和测试集中都有分布。

        Args:
            test_size: 测试集比例，默认为 0.2。

        Returns:
            包含元数据、训练集和测试集的字典。

        Raises:
            ValueError: 当某受试者文件数少于 2 时抛出。
        """
        subject_to_samples = self._group_by_subject()

        # 检查数据充足性
        insufficient_subjects = [
            subject
            for subject, samples in subject_to_samples.items()
            if len(samples) < 2
        ]
        if insufficient_subjects:
            raise ValueError(
                f"以下受试者文件数少于 2，无法进行划分: {insufficient_subjects}"
            )

        test_samples_flat: List[dict] = []
        train_samples_flat: List[dict] = []

        for subject, samples in subject_to_samples.items():
            # 创建副本以避免修改原列表，并打乱顺序
            shuffled_samples = np.asarray(samples.copy(), dtype=object)
            self.rng.shuffle(shuffled_samples)

            # 计算测试文件数：至少 1 个，最多 len-1 个
            test_count = max(
                1,
                min(
                    len(shuffled_samples) - 1,
                    round(len(shuffled_samples) * test_size),
                ),
            )
            
            test_samples_flat.extend(
                [self._create_sample_payload(s) for s in shuffled_samples[:test_count]]
            )
            train_samples_flat.extend(
                [self._create_sample_payload(s) for s in shuffled_samples[test_count:]]
            )

        # 构建元数据
        meta = {
            "seed": self.seed,
            "test_size_requested": test_size,
            "test_size_effective": len(test_samples_flat) / max(1, len(self.samples)),
            "split_strategy": "subject_stratified_file_split",
            "num_train_files": len(train_samples_flat),
            "num_test_files": len(test_samples_flat),
            "num_total_files": len(self.samples),
            "test_subject_distribution": self._count_subject_distribution(
                test_samples_flat
            ),
        }

        output = {
            "meta": meta,
            "train": train_samples_flat,
            "test": test_samples_flat,
        }
        self._save_split(output, "rssi_split_classification.pkl")

        return output

    def split_identification(
        self,
        test_size: float = 0.2,
        unknown_ratio: float = 0.4,
        max_unknown_files: Optional[int] = None,
        allow_unknown_sampling: bool = False,
        selection_strategy: Literal[
            "file_count", "subject_count", "balanced"
        ] = "file_count",
    ) -> dict:
        """识别任务：开集 1-vs-N 模拟划分。

        优化策略：
        1. 支持按文件数或人数控制未知身份比例，避免测试集过度膨胀。
        2. 可限制未知身份最大文件数，保持训练集规模。
        3. 支持对未知受试者文件进行抽样，而非全部使用。
        4. 提供多种未知身份选择策略。

        Args:
            test_size: 已知身份内部的测试比例，默认为 0.2。
            unknown_ratio: 未知身份目标占比，默认为 0.4。
            max_unknown_files: 未知身份最大文件数，None 表示不限制。
            allow_unknown_sampling: 是否允许对未知受试者文件进行抽样。
            selection_strategy: 未知身份选择策略。
                - "file_count": 按文件数比例选择未知受试者。
                - "subject_count": 按人数比例选择未知受试者。
                - "balanced": 平衡文件数和人数。

        Returns:
            包含元数据、训练集和测试集的字典。

        Raises:
            ValueError: 当受试者数量不足或文件数不足时抛出。
        """
        subject_to_samples = self._group_by_subject()
        subjects = sorted(subject_to_samples.keys())
        total_files = len(self.samples)
        total_subjects = len(subjects)

        if total_subjects < 3:
            raise ValueError(
                f"受试者数量 {total_subjects} 少于 3，无法执行开集划分"
            )

        # 计算每个受试者的文件数
        subject_file_counts = {
            subject: len(samples) for subject, samples in subject_to_samples.items()
        }

        # 根据策略选择未知受试者
        unknown_subjects = self._select_unknown_subjects(
            subjects=subjects,
            subject_file_counts=subject_file_counts,
            unknown_ratio=unknown_ratio,
            selection_strategy=selection_strategy,
            max_unknown_files=max_unknown_files,
        )

        # 确保已知受试者至少 2 个
        known_subjects = set(subjects) - unknown_subjects
        if len(known_subjects) < 2:
            # 从未知受试者中移出文件数最多的到已知
            sorted_unknown = sorted(
                unknown_subjects,
                key=lambda s: subject_file_counts[s],
                reverse=True,
            )
            for subject in sorted_unknown:
                if len(known_subjects) >= 2:
                    break
                unknown_subjects.remove(subject)
                known_subjects.add(subject)

        # 验证已知受试者文件数充足性
        insufficient_known = [
            subject for subject in known_subjects
            if len(subject_to_samples[subject]) < 2
        ]

        if insufficient_known:
            raise ValueError(
                f"已知受试者 {insufficient_known} 文件数少于 2，"
                f"无法进行训练/测试划分。请调整未知身份选择策略或增加数据。"
            )

        # 划分训练集和测试集
        train_samples_flat: List[dict] = []
        known_test_samples_flat: List[dict] = []

        for subject in known_subjects:
            subject_samples = subject_to_samples[subject]
            indices = np.arange(len(subject_samples))
            self.rng.shuffle(indices)

            # 计算该受试者的测试文件数
            test_count = max(
                1,
                min(
                    len(subject_samples) - 1,
                    round(len(subject_samples) * test_size),
                ),
            )
            test_indices_set = set(indices[:test_count].tolist())

            for idx, sample in enumerate(subject_samples):
                payload = self._create_sample_payload(sample)
                if idx in test_indices_set:
                    known_test_samples_flat.append(payload)
                else:
                    train_samples_flat.append(payload)

        # 处理未知受试者文件
        unknown_test_samples_flat = self._prepare_unknown_samples(
            unknown_subjects=unknown_subjects,
            subject_to_samples=subject_to_samples,
            max_unknown_files=max_unknown_files,
            allow_sampling=allow_unknown_sampling,
        )

        test_samples_flat = known_test_samples_flat + unknown_test_samples_flat

        # 构建元数据
        meta = {
            "seed": self.seed,
            "test_size_requested": test_size,
            "test_size_effective": len(test_samples_flat) / max(1, total_files),
            "split_strategy": "open_set_known_unknown_split",
            "selection_strategy": selection_strategy,
            "num_train_files": len(train_samples_flat),
            "num_test_files": len(test_samples_flat),
            "num_total_files": total_files,
            "known_subjects": sorted(known_subjects),
            "unknown_subjects": sorted(unknown_subjects),
            "known_test_files": len(known_test_samples_flat),
            "unknown_test_files": len(unknown_test_samples_flat),
            "unknown_file_ratio": len(unknown_test_samples_flat) / max(1, total_files),
            "subject_file_distribution": subject_file_counts,
        }

        output = {
            "meta": meta,
            "train": train_samples_flat,
            "test": test_samples_flat,
        }
        self._save_split(output, "rssi_split_identification.pkl")

        return output

    def _select_unknown_subjects(
        self,
        subjects: List[str],
        subject_file_counts: Dict[str, int],
        unknown_ratio: float,
        selection_strategy: Literal["file_count", "subject_count", "balanced"],
        max_unknown_files: Optional[int] = None,
    ) -> Set[str]:
        """根据策略选择未知受试者。

        Args:
            subjects: 所有受试者列表。
            subject_file_counts: 每个受试者的文件数映射。
            unknown_ratio: 未知身份目标占比。
            selection_strategy: 选择策略。
            max_unknown_files: 未知身份最大文件数。

        Returns:
            未知受试者集合。
        """
        total_files = sum(subject_file_counts.values())
        total_subjects = len(subjects)

        # 计算目标未知文件数
        target_unknown_files = int(round(total_files * unknown_ratio))
        if max_unknown_files is not None:
            target_unknown_files = min(target_unknown_files, max_unknown_files)

        # 计算目标未知人数
        target_unknown_subjects = max(1, int(round(total_subjects * unknown_ratio)))

        # 按文件数排序（优先选择文件数较少的受试者作为未知身份，增加难度多样性）
        sorted_by_files = sorted(
            subjects,
            key=lambda s: (subject_file_counts[s], self.rng.random()),
        )

        unknown_subjects: Set[str] = set()
        unknown_file_count = 0

        if selection_strategy == "file_count":
            # 按文件数比例选择
            for subject in sorted_by_files:
                if unknown_file_count >= target_unknown_files:
                    break
                # 确保不会因为添加该受试者而严重超过目标
                if (
                    unknown_file_count + subject_file_counts[subject]
                    <= target_unknown_files * 1.2
                ):
                    unknown_subjects.add(subject)
                    unknown_file_count += subject_file_counts[subject]

        elif selection_strategy == "subject_count":
            # 按人数比例选择
            shuffled = np.asarray(subjects, dtype=object)
            self.rng.shuffle(shuffled)
            unknown_subjects = set(shuffled[:target_unknown_subjects].tolist())

        elif selection_strategy == "balanced":
            # 平衡策略：综合考虑文件数和人数
            max_files = max(subject_file_counts.values()) if subject_file_counts else 1
            scores = {
                subject: 1.0 - (count / max_files) * 0.5
                for subject, count in subject_file_counts.items()
            }

            # 按得分排序
            sorted_by_score = sorted(
                subjects,
                key=lambda s: (scores[s], self.rng.random()),
                reverse=True,
            )

            for subject in sorted_by_score:
                if len(unknown_subjects) >= target_unknown_subjects:
                    break
                if unknown_file_count >= target_unknown_files:
                    break
                if (
                    unknown_file_count + subject_file_counts[subject]
                    <= target_unknown_files * 1.3
                ):
                    unknown_subjects.add(subject)
                    unknown_file_count += subject_file_counts[subject]

        # 确保至少选择 1 个未知受试者
        if len(unknown_subjects) == 0 and subjects:
            unknown_subjects.add(sorted_by_files[0])

        return unknown_subjects

    def _prepare_unknown_samples(
        self,
        unknown_subjects: Set[str],
        subject_to_samples: Dict[str, List[Sample]],
        max_unknown_files: Optional[int],
        allow_sampling: bool,
    ) -> List[dict]:
        """准备未知受试者的测试样本。

        Args:
            unknown_subjects: 未知受试者集合。
            subject_to_samples: 受试者到其 Sample 列表的映射。
            max_unknown_files: 未知身份最大文件数。
            allow_sampling: 是否允许抽样。

        Returns:
            未知测试样本列表。
        """
        unknown_test_samples_flat: List[dict] = []

        # 收集所有未知受试者的样本
        all_unknown_samples: List[Sample] = []
        for subject in unknown_subjects:
            all_unknown_samples.extend(subject_to_samples[subject])

        total_unknown_files = len(all_unknown_samples)

        # 检查是否需要限制文件数
        if (
            max_unknown_files is not None
            and total_unknown_files > max_unknown_files
            and total_unknown_files > 0
        ):
            if allow_sampling:
                # 随机抽样
                sampled_indices = self.rng.choice(
                    total_unknown_files,
                    size=max_unknown_files,
                    replace=False,
                )
                for idx in sampled_indices:
                    sample = all_unknown_samples[int(idx)]
                    unknown_test_samples_flat.append(self._create_sample_payload(sample))
            else:
                # 按受试者均匀选择
                files_per_subject = max(1, max_unknown_files // len(unknown_subjects))
                for subject in unknown_subjects:
                    subject_samples = subject_to_samples[subject]
                    if len(subject_samples) > files_per_subject:
                        sampled_indices = self.rng.choice(
                            len(subject_samples),
                            size=files_per_subject,
                            replace=False,
                        )
                        for idx in sampled_indices:
                            unknown_test_samples_flat.append(
                                self._create_sample_payload(subject_samples[int(idx)])
                            )
                    else:
                        for sample in subject_samples:
                            unknown_test_samples_flat.append(
                                self._create_sample_payload(sample)
                            )
        else:
            # 全部使用
            for subject in unknown_subjects:
                for sample in subject_to_samples[subject]:
                    unknown_test_samples_flat.append(self._create_sample_payload(sample))

        return unknown_test_samples_flat


# =============================================================================
# 便捷函数
# =============================================================================

def split_classification_dataset(
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """执行分类任务数据划分。

    Args:
        test_size: 测试集比例。
        seed: 随机种子。

    Returns:
        划分结果字典。
    """
    splitter = DatasetSplitter(seed=seed)
    return splitter.split_classification(test_size=test_size)


def split_identification_dataset(
    test_size: float = 0.2,
    seed: int = 42,
    unknown_ratio: float = 0.4,
    max_unknown_files: Optional[int] = None,
    allow_unknown_sampling: bool = False,
    selection_strategy: Literal[
        "file_count", "subject_count", "balanced"
    ] = "file_count",
) -> dict:
    """执行识别任务数据划分。

    Args:
        test_size: 已知身份内部的测试集比例。
        seed: 随机种子。
        unknown_ratio: 未知身份占比。
        max_unknown_files: 未知身份最大文件数。
        allow_unknown_sampling: 是否允许对未知受试者文件进行抽样。
        selection_strategy: 未知身份选择策略。

    Returns:
        划分结果字典。
    """
    splitter = DatasetSplitter(seed=seed)
    return splitter.split_identification(
        test_size=test_size,
        unknown_ratio=unknown_ratio,
        max_unknown_files=max_unknown_files,
        allow_unknown_sampling=allow_unknown_sampling,
        selection_strategy=selection_strategy,
    )


# =============================================================================
# 命令行入口
# =============================================================================

def main() -> None:
    """主函数：解析命令行参数并执行数据集划分。"""
    parser = argparse.ArgumentParser(description="RSSI 数据集划分工具")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例，默认 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--unknown-ratio",
        type=float,
        default=0.4,
        help="开集识别中未知身份占比，默认 0.4",
    )
    parser.add_argument(
        "--max-unknown-files",
        type=int,
        default=None,
        help="未知身份最大文件数，默认不限制",
    )
    parser.add_argument(
        "--allow-unknown-sampling",
        action="store_true",
        help="允许对未知受试者文件进行抽样",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="file_count",
        choices=["file_count", "subject_count", "balanced"],
        help="未知身份选择策略，默认 file_count",
    )
    args = parser.parse_args()
    
    splitter = DatasetSplitter(seed=args.seed)

    # 执行分类任务划分
    class_result = splitter.split_classification(test_size=args.test_size)
    print(
        f"分类任务划分完成: "
        f"训练集 {class_result['meta']['num_train_files']} 文件, "
        f"测试集 {class_result['meta']['num_test_files']} 文件"
    )

    # 执行识别任务划分
    id_result = splitter.split_identification(
        test_size=args.test_size,
        unknown_ratio=args.unknown_ratio,
        max_unknown_files=args.max_unknown_files,
        allow_unknown_sampling=args.allow_unknown_sampling,
        selection_strategy=args.selection_strategy,  # type: ignore
    )
    print(
        f"识别任务划分完成: "
        f"训练集 {id_result['meta']['num_train_files']} 文件, "
        f"测试集 {id_result['meta']['num_test_files']} 文件 "
        f"(已知 {id_result['meta']['known_test_files']}, "
        f"未知 {id_result['meta']['unknown_test_files']})"
    )


if __name__ == "__main__":
    main()