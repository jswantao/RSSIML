# RSSIML 项目代码审查报告

> 审查日期: 2026-05-18  
> 项目: WiFiML — 基于 Wi-Fi 信号的身份认证系统  
> 审查范围: 全部 Python 源码 (8411 行) + CI/CD + 依赖管理

---

## 📋 审查摘要

| 严重性 | 数量 | 说明 |
|:------:|:----:|------|
| 🔴 **致命** | 5 | 导致项目完全无法运行的 import 错误 |
| 🟠 **严重** | 4 | 函数签名不匹配、CI/CD 无法执行 |
| 🟡 **中等** | 5 | 代码不一致、潜在运行时错误 |
| 🟢 **建议** | 6 | 代码质量、依赖管理、架构优化 |

---

## 🔴 致命问题 (项目无法运行)

### 1. `scripts.models` 模块不存在

**影响范围**: `scripts/__init__.py`, `scripts/pipeline_runner.py`, `app_auth.py`, `experiments/base.py`, `experiments/e2_rssi.py`, `experiments/e4_compare.py`, `experiments/e5_rssi_slice.py`

```python
# scripts/__init__.py (第 51 行)
from scripts.models import (
    CNNConfig, CNNTrainConfig, SVMConfig,
    AuthenticationModel, CNNAuthenticationModel, RSSICNNBinaryClassifier,
    CNNInference, CNNTrainer, SVMAuthenticationTrainer, TrainingError,
    MetricsCalculator, compute_threshold, evaluate_authentication,
    svm_scores, load_cnn_checkpoint, train_cnn_authentication,
)

# scripts/pipeline_runner.py (第 46 行)
from scripts.models import (
    CNNTrainConfig, CNNTrainer, get_memory_monitor,
    SVMConfig, SVMAuthenticationTrainer, clear_gpu_memory, log_training,
)

# app_auth.py (第 38, 44 行)
from scripts.models import (CNNInference, clear_gpu_memory, log_training, svm_scores,)
from scripts.models.memory import get_memory_monitor
```

**问题**: 整个 `scripts/models/` 包完全缺失。这是一个涵盖 SVM 训练器、CNN 模型定义、推理引擎、评估指标计算等核心功能的模块包。项目中至少有 **7 个文件** 依赖此模块，项目当前无法通过 `import scripts` 导入。

**修复**: 需要创建完整的 `scripts/models/` 包,包含以下子模块:
- `scripts/models/__init__.py` — 导出所有公共 API
- `scripts/models/svm.py` — `SVMConfig`, `SVMAuthenticationTrainer`, `AuthenticationModel`, `svm_scores`, `compute_threshold`
- `scripts/models/cnn/` — CNN 相关 (`CNNConfig`, `CNNTrainConfig`, `CNNTrainer`, `CNNInference`, `RSSICNNBinaryClassifier`, `CNNAuthenticationModel`)
- `scripts/models/metrics.py` — `MetricsCalculator`, `evaluate_authentication`
- `scripts/models/memory.py` — `get_memory_monitor`, `clear_gpu_memory`
- `scripts/models/logging.py` — `log_training`

---

### 2. `PipelineConfig.subject_map()` / `subject_unmap()` 方法不存在

**影响范围**: `scripts/split_rssi_dataset.py` (第 201, 210 行), `experiments/base.py` (第 206 行), `app_auth.py` (第 856, 907 行)

```python
# split_rssi_dataset.py (第 201 行)
subj_map = self.config.subject_map("rssi")

# experiments/base.py (第 206 行)
unmap = self._pcfg.subject_unmap("rssi")

# app_auth.py (第 856 行)
fwd = _CONFIG.subject_map(source)
```

**问题**: `PipelineConfig` 类中只定义了 `SUBJECT_MAP_CSI` 和 `SUBJECT_MAP_RSSI` 两个 **类变量** (ClassVar)，但并没有定义 `subject_map(source)` 和 `subject_unmap(source)` **方法**。多个关键模块调用这些不存在的方法，会导致 `AttributeError`。

**修复**: 在 `PipelineConfig` 中添加:

```python
def subject_map(self, source: str) -> dict[str, str]:
    """获取用户 ID 映射 (原始 → 标准化)。"""
    if source == "csi":
        return self.SUBJECT_MAP_CSI
    return self.SUBJECT_MAP_RSSI

def subject_unmap(self, source: str) -> dict[str, str]:
    """获取用户 ID 反向映射 (标准化 → 原始)。"""
    fwd = self.subject_map(source)
    return {v: k for k, v in fwd.items()}
```

---

### 3. `_clear_sample_cache` 函数不存在

**影响范围**: `scripts/pipeline_runner.py` (第 44, 739 行)

```python
# pipeline_runner.py (第 44 行)
from scripts.split_rssi_dataset import DatasetSplitter, _clear_sample_cache

# pipeline_runner.py (第 739 行)
_clear_sample_cache(self._p._load_ctx); gc.collect()
```

**问题**: `scripts/split_rssi_dataset.py` 中不存在 `_clear_sample_cache` 函数。代码重构时此函数被移除但引用未清理。

**修复**: 在 `scripts/split_rssi_dataset.py` 中添加:

```python
def _clear_sample_cache(context: DataLoadContext | None = None) -> None:
    """清除样本缓存 — 释放 DataLoadContext 持有的数据。"""
    if context is not None:
        context.clear()
```

---

### 4. `scripts.train_cnn_models` 模块不存在

**影响范围**: `tests/test_cnn_mock.py`

```python
from scripts.train_cnn_models import RSSICNNClassifier, predict_cnn_probabilities, predict_cnn_windows
```

**问题**: 测试文件引用的是旧版模块 `scripts.train_cnn_models`，该模块已在重构时被移除/合并到 `scripts.models` 中，但测试未同步更新。

---

### 5. `process_features_pca_norm` 旧接口不存在

**影响范围**: `tests/test_feature_mock.py`

```python
features = feature_module.extract_frequency_domain_features(windows, low_freq_bins=12)
result = feature_module.process_features(windowed_file=..., output_file=..., ...)
```

**问题**: 测试文件引用的 `extract_frequency_domain_features()` 和 `process_features()` 是旧版函数接口，当前版本已重构为 `FeatureExtractor` 类，这些函数不再存在。

---

## 🟠 严重问题 (功能错误)

### 6. `save_experiment_subfigures()` 函数签名不匹配

**影响范围**: `experiments/e1_csi.py`, `e2_rssi.py`, `e3_enrollment.py`, `e5_rssi_slice.py`, `app_auth.py`

```python
# 函数签名需要 3 个参数:
def save_experiment_subfigures(fig, exp_name, output_dir):
    ...

# 但所有调用只传了 2 个参数:
save_experiment_subfigures(fig, "E1")   # 缺少 output_dir!
save_experiment_subfigures(fig, "E2")
save_experiment_subfigures(fig, "E3")
save_experiment_subfigures(fig, "E5")
_save_experiment_subfigures(fig, "E4")
```

**问题**: `output_dir` 参数没有默认值，所有调用都缺少该参数，运行时会抛出 `TypeError: save_experiment_subfigures() missing 1 required positional argument: 'output_dir'`。

**修复**: 为 `output_dir` 添加默认值:

```python
def save_experiment_subfigures(
    fig: plt.Figure, 
    exp_name: str, 
    output_dir: Path | str = "results/figures",
) -> list[Path]:
```

---

### 7. GitHub Actions CI/CD YAML 语法错误

**文件**: `.github/workflows/python-app.yml`

```yaml
于:           # ← 应为 "on:"
  push:
    分支: [ "main" ]    # ← 应为 "branches:"
  pull_request:
    分支: [ "main" ]    # ← 应为 "branches:"
```

**问题**: YAML 中使用了中文关键字 `于:` 和 `分支:`，GitHub Actions 无法识别，CI/CD 流水线完全不会触发。

---

### 8. `save_experiment_subfigures` 中 suptitle 恢复逻辑 bug

```python
# scripts/app_utils.py 第 125-126 行
restores[-1] = stitle    # 存的是 suptitle 对象

# 恢复时 (第 136 行)
for j, visible in restores.items():
    if j == -1:
        visible.set_visible(True)  # visible 是 suptitle 对象, 调用正确
    else:
        fig.axes[j].set_visible(visible)  # visible 是 bool, 正确
```

**问题**: 虽然当 `j == -1` 时 `visible` 确实是 suptitle 对象可以调用 `set_visible(True)`，但逻辑不清晰且有隐患 — 当 suptitle 原本就是不可见的情况下会被错误地恢复为可见。应该保存原始可见状态。

---

### 9. `csi_signal_viz.py` 和 `csi_feature_viz.py` 字号配置不一致

```python
# csi_signal_viz.py (独立的局部变量)
_FONT = {"small": 9, "normal": 11, "large": 13, "title": 15}

# csi_feature_viz.py (独立的局部变量)
_FONT = {"small": 9, "normal": 11, "large": 13, "title": 15}

# app_utils.py (统一的全局配置)
FONT_SIZES = {"small": 14, "normal": 15, "large": 16, "title": 18}
```

**问题**: 两个可视化脚本各自定义了独立的字号配置 `_FONT`，与 `app_utils.py` 中统一的 `FONT_SIZES` 不一致 (9/11/13/15 vs 14/15/16/18)。README 明确提到"统一字号配置 FONT_SIZES"，但这两个脚本并未遵循。

---

## 🟡 中等问题

### 10. `csi_signal_viz.py` 和 `csi_feature_viz.py` 重复 `load_combined_csi` 实现

两个可视化脚本各自完整实现了 `load_combined_csi()`、`butter_lowpass()`、`_save_subplot()`、`_setup_style()` 等函数，代码几乎完全相同 (~50 行重复)。违反 DRY 原则，后续维护时容易产生不一致。

**建议**: 将共享逻辑提取到 `scripts/app_utils.py` 或新建 `scripts/csi_utils.py`。

---

### 11. `FeatureExtractor` 缓存的线程安全问题

```python
# app_utils.py
def get_auth_feature_extractor(cfg: PreprocessConfig) -> FeatureExtractor:
    key = hash(cfg)
    if key not in _AUTH_FE_CACHE:
        with _CACHE_LOCK:
            if key not in _AUTH_FE_CACHE:
                _AUTH_FE_CACHE[key] = FeatureExtractor(cfg)
    return _AUTH_FE_CACHE[key]
```

然后在 `extract_features_for_auth()` 中:

```python
fe = get_auth_feature_extractor(cfg)
fe.pca = pca       # ← 直接修改共享实例的可变状态!
fe.scaler = scaler  # ← 在并发场景下会互相覆盖!
```

**问题**: 从缓存获取的 `FeatureExtractor` 实例是全局共享的，但随即直接修改其 `pca` 和 `scaler` 属性。如果两个线程同时对不同用户执行认证，会互相覆盖 pca/scaler，导致特征提取错误。

**修复**: 缓存应该只缓存配置相关的提取器，pca/scaler 应通过参数传递而非修改实例属性。

---

### 12. `save_experiment_figure()` 函数从未被使用

`app_utils.py` 中定义了 `save_experiment_figure()` 函数（保存整张图），但项目中所有调用都是 `save_experiment_subfigures()`（保存各子图），前者成为死代码。

---

### 13. E2/E5 实验中的循环内部重复导入

```python
# experiments/e2_rssi.py (第 53-54 行, 在 for 循环体内)
from scripts.app_utils import build_windows, extract_features_for_auth
from scripts.models import svm_scores

# experiments/e5_rssi_slice.py (类似)
from scripts.app_utils import slice_rssi, build_windows, extract_features_for_auth
```

**问题**: 这些 import 语句放在了逐用户的循环体内部，虽然 Python 的 import 系统会缓存已导入的模块，但这是不良的编码实践，增加了代码阅读的困惑度。

---

### 14. `pipeline_runner.py` 使用了 `Optional` 和 `Callable` 旧式类型注解

```python
from typing import Callable, Literal, Optional, Any

# 但项目文档声明"全面采用 Python 3.10+ 类型注解"
# 应该使用 X | None 代替 Optional[X]
```

**问题**: 与其他模块 (config.py, data_loader.py 等) 的 `X | None` 风格不一致。

---

## 🟢 建议

### 15. `requirements.txt` 包含未使用的依赖

`xgboost>=2.0.0` 和 `lightgbm>=4.0.0` 在项目中**完全未被使用**（grep 搜索结果为空），但作为重型依赖占用安装时间和空间。

### 16. `requirements.txt` 缺少实际使用的依赖

`psutil` 在 `data_loader.py` 中使用 (`import psutil`)，但未列入 `requirements.txt`。

### 17. 测试覆盖率极低

仅有 2 个测试文件 (117 行)，且都引用已不存在的旧版 API，**100% 的现有测试无法运行**。整个项目 8400+ 行代码实际零测试覆盖。

### 18. `app_auth.py` 超长单文件

`app_auth.py` 有 **2069 行**，包含了 UI 渲染、训练逻辑、推理流程、实验管理等多种职责。建议拆分为:
- `app_auth.py` — 主入口 + 路由
- `app_training.py` — 训练界面
- `app_inference.py` — 推理界面  
- `app_experiments.py` — 实验界面

### 19. 可视化脚本使用硬编码路径

```python
# csi_signal_viz.py
_CSI_DIR = Path("WiFi")
_OUT_DIR = Path("results/figures")
```

这些路径应从 `PipelineConfig` 获取，以保持与全局配置的一致性。

### 20. `ExperimentRunner` 的混入 (Mixin) 模式不规范

```python
class ExperimentRunner(BaseExperimentRunner):
    run_e1 = run_e1  # 直接赋值模块级函数为类方法
    run_e2 = run_e2
    ...
```

这种模式虽然可行，但不符合 Python 类的惯用设计。建议使用标准的方法定义或 Mixin 类。

---

## 📊 问题严重性分布

```
致命 (🔴):  ████████████ 5 个 — 项目无法 import/运行
严重 (🟠):  ████████ 4 个 — 关键功能异常
中等 (🟡):  ██████████ 5 个 — 代码质量/一致性
建议 (🟢):  ████████████ 6 个 — 优化改进空间
```

**总结**: 项目架构设计合理 (数据加载→窗口构建→特征提取→模型训练→推理评估)，代码风格统一，文档详尽。但经历了一次大规模重构后，新旧接口之间出现了严重的断裂：核心模块 `scripts.models` 整个缺失，多个方法/函数被引用但未定义，所有测试都指向已废弃的旧版 API。**在修复 🔴 和 🟠 级别问题前，项目完全无法运行。**
