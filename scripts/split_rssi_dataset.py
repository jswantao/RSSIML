"""RSSI 数据集划分模块。

提供分类任务和开集识别任务的数据集划分功能，支持按人员分层的文件粒度划分。
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from scripts.config import PipelineConfig
from scripts.data_loader import load_rssi_data


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
    """
    
    def __init__(self, seed: int = 42) -> None:
        """初始化数据集划分器。
        
        Args:
            seed: 随机种子，用于结果复现。
        """
        self.config = PipelineConfig.from_root()
        self.samples = self._load_samples()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
    def _load_samples(self) -> List[Sample]:
        """加载所有 RSSI 数据样本。
        
        Returns:
            Sample 对象列表。
        """
        data = load_rssi_data(self.config.raw_dir)
        return [
            Sample(
                file_name=f"wipin_{subject}{session}.mat",
                subject=subject,
                session=session,
                data=matrix,
            )
            for subject, session, matrix in data
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
    
    def _group_by_subject(self) -> Dict[str, List[int]]:
        """按受试者分组样本索引。
        
        Returns:
            受试者到样本索引列表的映射字典。
        """
        groups: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            groups.setdefault(sample.subject, []).append(idx)
        return groups
    
    def _create_sample_dict(self, sample: Sample) -> dict:
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
        subject_to_indices = self._group_by_subject()
        
        # 为每个受试者分配测试集索引
        test_indices: Set[int] = set()
        for subject, indices in subject_to_indices.items():
            if len(indices) < 2:
                raise ValueError(f"受试者 {subject} 文件数少于 2，无法划分")
            
            shuffled = np.asarray(indices, dtype=int)
            self.rng.shuffle(shuffled)
            
            test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_size))))
            test_indices.update(shuffled[:test_count].tolist())
        
        # 构建训练集和测试集
        train_samples = []
        test_samples = []
        
        for idx, sample in enumerate(self.samples):
            payload = self._create_sample_dict(sample)
            if idx in test_indices:
                test_samples.append(payload)
            else:
                train_samples.append(payload)
        
        # 构建元数据
        meta = {
            "seed": self.seed,
            "test_size_requested": test_size,
            "test_size_effective": len(test_samples) / max(1, len(self.samples)),
            "split_strategy": "subject_stratified_file_split",
            "num_train_files": len(train_samples),
            "num_test_files": len(test_samples),
            "num_total_files": len(self.samples),
            "test_subject_distribution": self._count_subject_distribution(test_samples),
        }
        
        output = {"meta": meta, "train": train_samples, "test": test_samples}
        self._save_split(output, "rssi_split_classification.pkl")
        
        return output
    
    def split_identification(
        self,
        test_size: float = 0.2,
        unknown_ratio: float = 0.4,
    ) -> dict:
        """识别任务：开集 1-vs-N 模拟划分。
        
        策略：
        1. 随机抽取部分受试者作为未知身份。
        2. 已知身份按文件做训练/测试划分。
        3. 测试集包含已知身份测试文件和全部未知身份文件。
        
        Args:
            test_size: 已知身份内部的测试比例，默认为 0.2。
            unknown_ratio: 未知身份占比，默认为 0.4。
            
        Returns:
            包含元数据、训练集和测试集的字典。
            
        Raises:
            ValueError: 当受试者数量不足或文件数不足时抛出。
        """
        # 按受试者分组样本对象
        subject_to_samples: Dict[str, List[Sample]] = {}
        for sample in self.samples:
            subject_to_samples.setdefault(sample.subject, []).append(sample)
        
        subjects = sorted(subject_to_samples.keys())
        if len(subjects) < 3:
            raise ValueError(f"受试者数量 {len(subjects)} 少于 3，无法执行开集划分")
        
        # 划分已知/未知受试者
        shuffled_subjects = np.asarray(subjects, dtype=object)
        self.rng.shuffle(shuffled_subjects)
        
        unknown_count = max(1, min(len(subjects) - 2, int(round(len(subjects) * unknown_ratio))))
        unknown_subjects = set(shuffled_subjects[:unknown_count].tolist())
        known_subjects = set(shuffled_subjects[unknown_count:].tolist())
        
        # 划分训练集和测试集
        train_samples = []
        known_test_samples = []
        unknown_test_samples = []
        
        for subject in known_subjects:
            subject_samples = subject_to_samples[subject]
            if len(subject_samples) < 2:
                raise ValueError(f"已知受试者 {subject} 文件数少于 2，无法划分")
            
            indices = np.arange(len(subject_samples))
            self.rng.shuffle(indices)
            
            test_count = max(1, min(len(subject_samples) - 1, int(round(len(subject_samples) * test_size))))
            test_idx = set(indices[:test_count].tolist())
            
            for idx, sample in enumerate(subject_samples):
                payload = self._create_sample_dict(sample)
                if idx in test_idx:
                    known_test_samples.append(payload)
                else:
                    train_samples.append(payload)
        
        # 未知受试者全部进入测试集
        for subject in unknown_subjects:
            for sample in subject_to_samples[subject]:
                unknown_test_samples.append(self._create_sample_dict(sample))
        
        test_samples = known_test_samples + unknown_test_samples
        
        # 构建元数据
        meta = {
            "seed": self.seed,
            "test_size_requested": test_size,
            "test_size_effective": len(test_samples) / max(1, len(self.samples)),
            "split_strategy": "open_set_known_unknown_split",
            "num_train_files": len(train_samples),
            "num_test_files": len(test_samples),
            "num_total_files": len(self.samples),
            "known_subjects": sorted(known_subjects),
            "unknown_subjects": sorted(unknown_subjects),
            "known_test_files": len(known_test_samples),
            "unknown_test_files": len(unknown_test_samples),
        }
        
        output = {"meta": meta, "train": train_samples, "test": test_samples}
        self._save_split(output, "rssi_split_identification.pkl")
        
        return output
    
    @staticmethod
    def _count_subject_distribution(samples: List[dict]) -> Dict[str, int]:
        """统计样本中受试者的分布。
        
        Args:
            samples: 样本字典列表。
            
        Returns:
            受试者到样本数量的映射字典。
        """
        distribution: Dict[str, int] = {}
        for sample in samples:
            subject = str(sample["subject"])
            distribution[subject] = distribution.get(subject, 0) + 1
        return dict(sorted(distribution.items()))


def main() -> None:
    """主函数：解析命令行参数并执行数据集划分。"""
    parser = argparse.ArgumentParser(description="RSSI 数据集划分工具")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--unknown-ratio", type=float, default=0.4, help="开集识别中未知身份占比，默认 0.4")
    args = parser.parse_args()
    
    splitter = DatasetSplitter(seed=args.seed)
    
    # 执行分类任务划分
    class_result = splitter.split_classification(test_size=args.test_size)
    print(f"分类任务划分完成: "
          f"训练集 {class_result['meta']['num_train_files']} 文件, "
          f"测试集 {class_result['meta']['num_test_files']} 文件")
    
    # 执行识别任务划分
    id_result = splitter.split_identification(
        test_size=args.test_size,
        unknown_ratio=args.unknown_ratio,
    )
    print(f"识别任务划分完成: "
          f"训练集 {id_result['meta']['num_train_files']} 文件, "
          f"测试集 {id_result['meta']['num_test_files']} 文件 "
          f"(已知 {id_result['meta']['known_test_files']}, "
          f"未知 {id_result['meta']['unknown_test_files']})")


if __name__ == "__main__":
    main()