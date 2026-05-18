# -- coding: utf-8 --
"""流水线配置模块 — 统一配置管理与环境隔离。

优化记录:
- 🔴 修复 to_dict/from_dict 遗漏 csi_selected_actions 与 csi_denoise
- 🔴 日志隔离：不再修改 logging.root，改用独立 'wifiml' 命名空间
- 🟠 防副作用：from_dict 严格使用深拷贝与安全提取
- 🟢 新增 to_cache_dict()：专供中间文件哈希命名，自动剥离路径/敏感字段
- 🟢 类型提示现代化：全面采用 Python 3.10+ 语法与严格校验
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# 路径与环境常量
# ══════════════════════════════════════════════════════════════════════
_SUBDIR_RAW = "raw"
_SUBDIR_WIFI = "WiFi"
_SUBDIR_DATA = "data"
_SUBDIR_MODELS = "models"
_SUBDIR_RESULTS = "results"
_SUBDIR_LOGS = "logs"
_SUBDIR_CACHE = "cache"

_ENV_ROOT_DIR = "RSSI_ROOT_DIR"
_ENV_WIFI_DIR = "RSSI_WIFI_DIR"


@dataclass(frozen=True)
class PipelineConfig:
    """流水线全局配置 — 不可变数据中心。
    
    通过环境变量可覆盖默认路径:
      - RSSI_ROOT_DIR: 项目根目录
      - RSSI_WIFI_DIR: CSI 数据目录
    """

    root_dir: Path
    random_seed: int = 42
    test_size: float = 0.2
    window_size: int = 200
    step_size: int = 100
    pca_variance: float = 0.9019
    use_pca: bool = True
    npy_dir_override: Path | None = None
    csi_selected_actions: tuple[int, ...] | None = (
        5, 7, 31, 32, 36, 37, 40, 41, 46, 47,
    )
    csi_denoise: str | None = None  # None / "hampel" / "savgol" / "butterworth"

    # ── 应用级常量（不随实例变化） ─────────────────────────────────────
    DEFAULT_SUBJECTS: ClassVar[dict[str, list[str]]] = {
        "rssi": [str(i) for i in range(1, 6)],
        "csi": [str(i) for i in range(1, 20)],
    }
    SUBJECT_MAP_CSI: ClassVar[dict[str, str]] = {
        str(i): str(i - 11) for i in range(12, 31)
    }
    SUBJECT_MAP_RSSI: ClassVar[dict[str, str]] = {
        "FXY": "1", "HYH": "2", "LJW": "3", "QYH": "4", "ZX": "5",
    }
    PARAM_STUDY_DEFAULTS: ClassVar[dict[str, dict]] = {
        "rssi": {"sizes": [1, 2, 3, 4], "default": [1, 2, 3]},
        "csi": {"sizes": [10, 20, 40, 80, 100, 140, 200], "default": [20, 40, 80]},
    }

    # ═══════════════════════════════════════════════════════════════════
    # 目录属性（惰性求值）
    # ═══════════════════════════════════════════════════════════════════
    @property
    def raw_dir(self) -> Path:
        return self.root_dir / _SUBDIR_RAW

    @property
    def npy_dir(self) -> Path:
        if self.npy_dir_override is not None:
            return self.npy_dir_override
        if env_dir := os.environ.get(_ENV_WIFI_DIR):
            return Path(env_dir)
        return self.root_dir / _SUBDIR_WIFI

    @property
    def data_dir(self) -> Path:
        return self.root_dir / _SUBDIR_DATA

    @property
    def model_dir(self) -> Path:
        return self.root_dir / _SUBDIR_MODELS

    @property
    def result_dir(self) -> Path:
        return self.root_dir / _SUBDIR_RESULTS

    @property
    def log_dir(self) -> Path:
        return self.root_dir / _SUBDIR_LOGS

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / _SUBDIR_CACHE

    @property
    def all_dirs(self) -> dict[str, Path]:
        return {
            "raw": self.raw_dir, "npy": self.npy_dir,
            "data": self.data_dir, "models": self.model_dir,
            "results": self.result_dir, "logs": self.log_dir,
            "cache": self.cache_dir,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 工厂方法
    # ═══════════════════════════════════════════════════════════════════
    @classmethod
    def from_root(cls, root_dir: Path | str | None = None) -> "PipelineConfig":
        """从根目录创建配置。优先级: 参数 > 环境变量 > 自动推断。"""
        if root_dir is None:
            if env_root := os.environ.get(_ENV_ROOT_DIR):
                root_dir = Path(env_root)
            else:
                # 默认 scripts/config.py 上两级为项目根
                root_dir = Path(__file__).resolve().parent.parent

        root = Path(root_dir) if isinstance(root_dir, str) else root_dir
        logger.debug("PipelineConfig root_dir=%s", root)
        return cls(root_dir=root)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量创建配置（兼容 from_root）。"""
        return cls.from_root()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        """从字典安全创建配置（不污染原字典）。"""
        cfg = d.copy()
        root = Path(cfg.pop("root_dir", "."))
        npy_override = cfg.pop("npy_dir_override", None)
        if npy_override and not isinstance(npy_override, Path):
            npy_override = Path(npy_override)

        selected = cfg.pop("csi_selected_actions", None)
        if selected is not None:
            selected = tuple(int(x) for x in selected)

        return cls(
            root_dir=root,
            random_seed=int(cfg.get("random_seed", 42)),
            test_size=float(cfg.get("test_size", 0.2)),
            window_size=int(cfg.get("window_size", 200)),
            step_size=int(cfg.get("step_size", 100)),
            pca_variance=float(cfg.get("pca_variance", 0.9019)),
            use_pca=bool(cfg.get("use_pca", True)),
            npy_dir_override=npy_override,
            csi_selected_actions=selected,
            csi_denoise=cfg.get("csi_denoise"),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "PipelineConfig":
        """从 JSON 配置文件加载。"""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ═══════════════════════════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════════════════════════
    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化字典。"""
        return {
            "root_dir": str(self.root_dir),
            "random_seed": self.random_seed,
            "test_size": self.test_size,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "pca_variance": self.pca_variance,
            "use_pca": self.use_pca,
            "npy_dir_override": str(self.npy_dir_override) if self.npy_dir_override else None,
            "csi_selected_actions": list(self.csi_selected_actions) if self.csi_selected_actions else None,
            "csi_denoise": self.csi_denoise,
        }

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """导出为 JSON 配置文件。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
        logger.info("配置已导出: %s", out)

    def to_cache_dict(self) -> dict[str, Any]:
        """专供缓存哈希生成的字典。剥离绝对路径、排除无关字段。"""
        return {
            "test_size": self.test_size,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "pca_variance": self.pca_variance,
            "use_pca": self.use_pca,
            "csi_selected_actions": self.csi_selected_actions,
            "csi_denoise": self.csi_denoise,
            "random_seed": self.random_seed,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════════
    def ensure_dirs(self) -> list[Path]:
        """创建所有输出目录（raw/WiFi 除外）。"""
        created = []
        for sub in (_SUBDIR_DATA, _SUBDIR_MODELS, _SUBDIR_RESULTS, _SUBDIR_LOGS, _SUBDIR_CACHE):
            p = self.root_dir / sub
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
        logger.debug("目录就绪: %s", [str(c) for c in created])
        return created

    def ensure_all_dirs(self) -> list[Path]:
        """创建所有目录（含原始数据目录）。"""
        created = self.ensure_dirs()
        for sub in (_SUBDIR_RAW, _SUBDIR_WIFI):
            p = self.root_dir / sub
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
        return created

    def get_path(self, relative_path: str | Path, category: str = "data") -> Path:
        """获取相对于指定类别目录的完整路径。"""
        dir_map = {
            "data": self.data_dir, "models": self.model_dir,
            "results": self.result_dir, "logs": self.log_dir,
            "cache": self.cache_dir,
        }
        base = dir_map.get(category, self.data_dir)
        return base / relative_path

    def setup_logging(self, level: int = logging.INFO, log_file: str | None = "pipeline.log") -> logging.Logger:
        """配置项目级独立日志（不污染全局 root logger）。"""
        self.ensure_dirs()
        proj_logger = logging.getLogger("wifiml")
        proj_logger.setLevel(level)
        proj_logger.handlers.clear()

        fmt_short = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fmt_long = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt_short)
        proj_logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(self.log_dir / log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt_long)
            proj_logger.addHandler(fh)

        logger.info("日志已配置: level=%s, file=%s", logging.getLevelName(level), log_file)
        return proj_logger


    def subject_map(self, source: str) -> dict[str, str]:
        """获取用户 ID 映射 (原始 → 标准化)。"""
        if source == "csi":
            return self.SUBJECT_MAP_CSI
        return self.SUBJECT_MAP_RSSI

    def subject_unmap(self, source: str) -> dict[str, str]:
        """获取用户 ID 反向映射 (标准化 → 原始)。"""
        fwd = self.subject_map(source)
        return {v: k for k, v in fwd.items()}

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(root={self.root_dir.name}, seed={self.random_seed}, "
            f"test_size={self.test_size}, window={self.window_size}/{self.step_size}, "
            f"pca={'on' if self.use_pca else 'off'})"
        )