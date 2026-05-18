# WiFiML — 基于 Wi-Fi 信号的身份认证系统

基于商用 Wi-Fi 信道状态信息 (CSI) 和接收信号强度指示 (RSSI) 的身份认证完整流水线，支持传统机器学习 (SVM/在线SVM) 与深度学习 (1D-CNN) 双路线，提供**单次认证**与**持续认证**两种推理范式，覆盖从原始数据加载到模型训练、评估、在线推理的端到端实现。

## 目录

- [认证机制](#认证机制)
  - [单次认证](#单次认证)
  - [持续认证](#持续认证)
  - [机制对比](#机制对比)
- [实验矩阵](#实验矩阵)
- [数据格式](#数据格式)
- [流水线架构](#流水线架构)
- [特征工程](#特征工程)
- [模型架构](#模型架构)
- [评估指标](#评估指标)
- [环境安装](#环境安装)
- [运行方式](#运行方式)
- [配置参数](#配置参数)
- [性能优化](#性能优化)
- [API 参考](#api-参考)
- [常见问题](#常见问题)

---

## 认证机制

本项目实现两种互补的身份认证范式——单次认证和持续认证。两者共享相同的**逐用户二分类验证** (Per-User Binary Verification) 训练框架，但在推理阶段的信号处理方式、决策逻辑和适用场景上存在根本性差异。

### 单次认证

**工作机制**：用户在系统前完成一次固定时长的 Wi-Fi 信号采集，系统对该样本做出一次性 **ACCEPT / REJECT** 判定。

**核心流程**：

```
上传样本 (MAT/NPY 或 CSI 组合拼接)
  │
  ├─ 标准模式: 全量信号 → build_windows(ws=200, ss=100) →
  │             逐窗口特征提取 → 验证器评分 →
  │             mean(所有窗口分数) ≥ 用户专属阈值 → ACCEPT / REJECT
  │
  └─ 切片模式 (RSSI): 长时样本 → slice_rssi(duration=5s) →
                       每片独立认证 (mean_score ≥ threshold) →
                       汇总: 通过率 ≥ 50% → ACCEPT
```

**决策规则**：

| 模式 | 判定逻辑 | 适用数据 |
|------|---------|---------|
| **标准** | `mean(scores) >= threshold` | CSI (~5s 短样本) |
| **切片** | `n_accept / n_total >= 0.5` | RSSI 长样本 (200s 切为 5s 片段) |
| **CSI 组合** | 同标准模式 | 拼接 10 动作 × 1 trial ≈ 50s |

**适用场景**：
- 设备解锁、门禁系统等一次性的身份核验
- 登录认证、支付确认等需要明确决策边界的场景
- 短时交互场景，用户完成固定动作后立即获得结果

**项目实现**：`app_auth.py` → `_render_single_auth()` (第 1539 行)，支持 SVM/CNN 双模型、文件上传与 CSI 动作组合模式。

### 持续认证

**工作机制**：系统持续采集用户的 Wi-Fi 信号，以滑动窗口方式逐段评分，通过前向平滑和时序分析追踪身份状态的动态变化，输出连续的 ACCEPT/REJECT 序列。

**核心流程**：

```
长时信号样本 (200s+)
  │
  ├─ [1] build_windows(raw, ws=200, ss=100) → 逐窗口评分
  │        scores[i] = verifier(features(window_i))
  │
  ├─ [2] 前向滑动平均 (因果平滑, 仅使用历史信息)
  │        smoothed[i] = mean(scores[max(0, i-ws+1) : i+1])
  │
  ├─ [3] 逐窗口二元判定
  │        decisions[i] = (smoothed[i] >= threshold)
  │
  └─ [4] 汇总指标
           ├─ 接受率: mean(decisions)
           ├─ 最长连续接受: 最大连续 True 长度
           ├─ 状态切换次数: decisions[i] != decisions[i-1] 的计数
           └─ 最终决策: decisions[-1] (当前时刻认证状态)
```

**前向滑动平均**：与普通移动平均不同，前向平滑仅使用当前及过去的窗口评分，不引入未来信息。这保证了认证系统在实时部署时的因果性——任意时刻的决策仅依赖已观测到的信号。

**状态切换检测**：当接受/拒绝状态频繁交替时 (`switches > 2`)，系统发出身份变化警告。这是持续认证区别于单次认证的关键能力——不仅能判断"当前是谁"，还能检测"身份何时发生变化"。

**适用场景**：
- 持续会话保护：用户在设备前持续工作时，后台静默监控身份一致性
- 异常行为检测：用户离开设备或换人操作时及时察觉
- 零交互认证：用户无需执行特定动作，日常行为即可维持认证状态
- 安全敏感环境：金融交易、机密文档操作等需要全程身份确认的场景

**项目实现**：`app_auth.py` → `_render_continuous_auth()` (第 1749 行)，核心评估逻辑复用于 `experiments/base.py` → `_compute_continuous_auth()` (第 84 行)。

### 机制对比

| 维度 | 单次认证 | 持续认证 |
|------|---------|---------|
| **输入** | 短时固定样本 (~5s CSI / 200s RSSI) | 长时连续信号 (200s+) |
| **输出** | 单个二元决策 (ACCEPT/REJECT) | 决策序列 + 时序统计 |
| **决策方式** | 全局平均分数 vs 阈值 | 逐窗口平滑分数 vs 阈值 |
| **时间建模** | 无 (空间聚合所有窗口) | 有 (前向滑动平均, 因果约束) |
| **状态切换检测** | 不支持 | 支持 (统计决策翻转次数) |
| **用户交互** | 需主动操作 (采集特定动作) | 可被动运行 (后台静默) |
| **延迟** | 采集完成后一次性判定 | 每个窗口均产生判定 |
| **可靠性** | 高 (更多信号 → 更稳定) | 逐窗口波动较大, 需平滑 |
| **典型数据** | CSI NPY (~1000 时间步) | RSSI MAT (~20000 时间步) |

**互补性**：单次认证提供高置信度的一次性身份确认，适合作为系统入口的安全闸门；持续认证提供时序细粒度的身份追踪，适合作为会话期间的持续保障。两者可组合使用——先通过单次认证登录，再通过持续认证监控会话安全。

---

## 实验矩阵

项目包含五个论文实验 (E1~E5)，系统验证认证系统的各项能力。所有实验通过 Streamlit 工作台的「🧪 实验」标签页一键运行，结果以论文级图表呈现。

| 编号 | 实验名称 | 数据源 | 模型 | 核心变量 | 验证能力 |
|:----:|---------|:-----:|:----:|---------|---------|
| **E1** | CSI 单次认证 | CSI | 在线 SVM (linear) | 文件级划分 | 单次认证精度上限、冒名攻击防御 |
| **E2** | RSSI 持续认证 | RSSI | SVM (RBF) | 全流程复现 | 持续认证可行性、逐窗口时序判定 |
| **E3** | 注册时长影响 | CSI | SVM | 训练样本数 20~200 | 最少注册样本量、学习曲线 |
| **E4** | 模型性能对比 | CSI | SVM vs CNN | 模型架构 + 速度 | 精度/速度权衡、选型依据 |
| **E5** | RSSI 切片单次认证 | RSSI | SVM (RBF) | 切片时长 3~20s | 切片粒度对判定可靠性的影响 |

### 各实验定位

**E1 — CSI 单次认证**：验证系统在理想条件下的认证精度上限。使用 CSI 高维度信号 (270 子载波)、文件级随机划分 (同动作训练/测试)、Butterworth 低通滤波降噪、在线 SVM linear 核。输出每用户 FRR/FAR 对比和冒名攻击防御评级，回答"CSI 能做到多准"。

**E2 — RSSI 持续认证**：全流程复现基础训练→推理阶段的持续认证方案。使用与推理阶段 `_render_continuous_auth` 完全一致的前向滑动平均 (窗口=10)、逐窗口判定、接受率/最长连续接受/状态切换等统计指标。回答"RSSI 持续认证是否可行"。

**E3 — 注册时长影响**：通过 6 组训练样本量 (20~200) 的学习曲线，确定达到可用认证精度所需的最少注册样本。输出 HTER/FAR/FRR/准确率随样本量的变化曲线和标准差置信带。回答"需要多少注册样本"。

**E4 — 模型性能对比**：SVM (在线 linear) vs CNN (轻量 1D-CNN) 在相同数据上的精度和训练耗时对比。SVM 使用 SGD+hinge loss 在线学习，CNN 使用 AMP+大 batch 速度优化 (关 checkpoint 提速 ~40%)。输出分组柱状图和耗时对比表。回答"SVM 和 CNN 选哪个"。

**E5 — RSSI 切片单次认证**：将长时 RSSI 样本按不同时长 (3~20s) 切片，每片独立执行单次认证判定 (与推理阶段切片模式一致)，汇总评估整体通过率。输出切片通过率-时长曲线、每用户分布和热力图。回答"多短的切片仍能可靠认证"。

### 数据流与运行方式

```
PipelineConfig
  ├─ csi_selected_actions = (5,7,31,32,36,37,40,41,46,47)
  ├─ SUBJECT_MAP_CSI  = {12→1, ..., 30→19}
  └─ SUBJECT_MAP_RSSI = {FXY→1, ..., ZX→5}

AuthPipeline(seed=42, window_size=200, step_size=100, ...)
  ├─ DatasetSplitter → 动作筛选 + 用户映射 + 文件级划分
  ├─ DataCoordinator.build_windows_memmap → 滑动窗口 + 可选降噪
  ├─ SVMAuthenticationTrainer → 在线 SVM (SGD+class_weight) / 批处理 SVC (RBF)
  └─ CNNTrainer → 1D-CNN (AMP + pos_weight + HTER 早停)
```

```bash
streamlit run app_auth.py   # 注册 → 🧪 实验 → 勾选 → 运行
```

多选支持，自动过滤与当前数据源不兼容的实验 (E2/E5 仅 RSSI，E1/E3 仅 CSI)。

---

## 数据格式

### RSSI 数据 (MAT 格式)

- **目录**: `raw/`
- **命名**: `wipin_<subject><session>.mat`
- **结构**: 每个 MAT 文件包含 `RSSI` 变量, 二维矩阵 `(time_steps, channels)`
- **规模**: 5 用户 × 4 会话 = 20 文件, ~200s/文件, 52 通道
- **加载**: `scipy.io.loadmat` → `np.float32` 转换

### CSI 数据 (NPY 格式)

- **目录**: `WiFi/`
- **命名**: `{subject}_{activity}_{trial}.npy`
- **结构**: `(subcarriers, time)` 形状, 加载时自动转置为 `(time, subcarriers)`
- **规模**: 19 用户 × 55 动作 × 20 重复 = 20,900 文件, ~5s/文件
- **子载波数**: 270 (80MHz 802.11ac)
- **时间步**: ~1000/文件
- **加载**: 元数据延迟加载 + LRU 缓存 + 异步批量 I/O, 避免内存溢出

### 精选动作集

从 55 个动作中筛选 10 个动作用于认证模型训练：

| ID | 动作 | 选择理由 |
|----|------|----------|
| 5 | 使用电话 | 日常高频动作 |
| 7 | 把某物放在桌子上 | 人-物交互、信号特征丰富 |
| 31 | 挥手 | 手部运动直接影响 WiFi 信号 |
| 32 | 拍手 | 短促动作、频率特征独特 |
| 36 | 坐着 | 身体姿态变化明显 |
| 37 | 站起来 | 姿态变化、特征丰富 |
| 40 | 伸展 | 全身运动 |
| 41 | 把手放在肩膀上 | 手部位置变化影响多径 |
| 46 | 摇头 | 头部运动影响信号传播路径 |
| 47 | 阅读 | 静态姿态、坐姿差异大 |

---

## 流水线架构

### 项目结构

```text
.
├── raw/                              # RSSI MAT 原始数据
├── WiFi/                             # CSI NPY 原始数据
├── data/                             # 中间数据 (划分、窗口、特征)
├── models/                           # 训练好的模型文件
├── results/                          # 评估结果 (JSON)
├── logs/                             # 训练日志 (JSONL)
├── cache/                            # 流水线缓存 (参数哈希命名, 支持断点续传)
├── scripts/
│   ├── __init__.py                   # 公共 API
│   ├── config.py                     # 统一配置管理
│   ├── data_loader.py                # MAT + NPY 数据加载 (LRU 缓存, 内存池, 异步批量)
│   ├── split_rssi_dataset.py         # 逐用户认证数据划分
│   ├── build_sliding_windows.py      # Stride-trick 零拷贝滑窗构建
│   ├── process_features_pca_norm.py  # 特征提取 + PCA + MinMax 归一化 + 降噪
│   ├── pipeline_runner.py            # 端到端认证流水线 (RSSI + CSI 统一)
│   ├── log_server.py                 # WebSocket 实时日志广播
│   └── models/
│       ├── base.py                   # 指标计算、阈值选择、认证评估
│       ├── config.py                 # SVM/CNN 配置 dataclass
│       ├── memory.py                 # GPU/系统内存监控与自动降级
│       ├── training_log.py           # 结构化 JSONL 训练日志
│       ├── svm/
│       │   ├── model.py              # AuthenticationModel 容器 (含 PCA/Scaler/feature_dim)
│       │   ├── trainers.py           # 批处理 + 在线 SVM 训练器
│       │   ├── utils.py              # svm_scores, 单用户验证器训练
│       │   └── online.py             # OnlineSVMVerifier (SGD + RBFSampler, float32 优化)
│       └── cnn/
│           ├── models.py             # ConvBackbone, BinaryClassifier, AuthModel
│           ├── trainer.py            # CNNTrainer (AMP + 负采样 + HTER 早停)
│           ├── inference.py          # CNNInference (按需 GPU 加载)
│           └── utils.py              # load_checkpoint, 训练便捷函数
├── experiments/                       # 论文实验模块 (E1~E4)
│   ├── __init__.py                    # 公共 API, ExperimentRunner 组合
│   ├── base.py                        # 共享基础设施 (流水线构建, 持续认证评估)
│   ├── e1_csi.py                      # E1: CSI 单次认证 (文件级划分)
│   ├── e2_rssi.py                     # E2: RSSI 持续认证 (推理阶段方案复现)
│   ├── e3_enrollment.py              # E3: 注册时长影响
│   ├── e4_compare.py                  # E4: 模型性能对比 (SVM vs CNN)
│   └── e5_rssi_slice.py               # E5: RSSI 切片式单次认证
├── scripts/
│   ├── app_utils.py                   # 共享工具 (plot样式, 特征提取, 信号切片)
├── app_auth.py                       # Streamlit 训练与推理工作台
└── requirements.txt
```

### 数据处理流程

```
raw/*.mat  或  WiFi/*.npy
  │
  ├─[1] data_loader.py               → 加载 + 校验 + 缓存 (LRU, 异步批量)
  ├─[2] split_rssi_dataset.py        → 动作筛选 → 用户编号映射 → 逐用户文件级划分
  ├─[3] build_sliding_windows.py     → Stride-trick 滑窗 (默认 200/100)
  ├─[4] CSI 原始信号降噪 (可选)       → Hampel / Savitzky-Golay / Butterworth
  ├─[5] process_features_pca_norm.py → 降噪 → 频域 + 统计 + 时域特征
  │                                     → PCA 降维 → MinMax 归一化
  └─[6] models/svm/ 或 models/cnn/   → 逐用户二分类验证器训练
                                        → 阈值选择 (Youden/EER/Quantile)
                                        → HTER/FAR/FRR 评估
```

### 数据划分策略

认证任务采用文件级认证划分：每个用户按文件级切分训练集和测试集，其中测试集包含该用户的 genuine 样本 (正类) 和其他用户的 impostor 样本 (负类)。CSI 支持跨动作划分 (`cross_activity=True`)，以某动作数据做训练、其他动作做测试，验证模型对未见动作的泛化能力。

中间文件以参数哈希命名，哈希基于 `test_size`, `random_seed`, `window_size`, `step_size`, `pca_variance`, `use_pca`, `feature_groups`, `max_files_per_subject`, `use_online_svm`, `online_kernel`, `csi_selected_actions`, `csi_denoise`, `_cache_fmt` 计算 MD5 摘要。不同参数自动产生独立缓存。

---

## 特征工程

特征提取流水线支持三组可配置的特征组，在滑动窗口的每个通道上独立提取后拼接：

### 特征组

| 特征组 | 每通道维度 | 描述 |
|--------|-----------|------|
| **spectral (频域)** | 16 | FFT 低频频点幅值, 捕获频域模式 |
| **statistical (统计)** | 4 | 标准差、偏度、峰度、过零率 |
| **temporal (时域)** | 9 | 自相关系数 (lag-1, lag-2)、差分统计 (均值, 标准差)、短时能量 (4 段)、信号熵 (10-bin Shannon) |

### 特征维度

| 数据源 | 通道数 | 全特征组维度 |
|--------|--------|-------------|
| RSSI | 52 | 52 × 29 = 1,508 |
| CSI | 270 | 270 × 29 = 7,830 |

### 预处理步骤

1. **CSI 原始信号降噪** (可选): Hampel (脉冲异常值去除) / Savitzky-Golay (多项式平滑) / Butterworth (低通, 20Hz 截止), 在窗口构建前对完整时间序列应用
2. **窗口内降噪** (可选): 移动平均 / 中值滤波 / 低通滤波, 对已构建窗口沿时间轴应用
3. **特征提取**: 对降噪后的窗口按启用的特征组提取特征 (fp32 精度)
4. **PCA 降维** (可选): 保留 90.19% 方差, `svd_solver='auto'`
5. **MinMax 归一化**: 缩放到 `[-1, 1]`

> CNN 模型直接使用降噪后的原始窗口数据, 不使用手工特征。

### 推理时特征自适应

训练时 `PreprocessConfig` (特征组、低频 bin 数、降噪配置) 随模型一同持久化。推理时 `_extract_features_for_auth` 自动读取模型携带的配置并验证特征维度一致性，提供精确的修复建议（如窗口大小调整）。

---

## 模型架构

### SVM — 逐用户二分类验证

- **批处理 SVM (RBF)**: `SVC(kernel='rbf', class_weight='balanced', C=svm_C, gamma=svm_gamma)`。当特征维度/样本数 > 0.5 时自动切换 `LinearSVC` 以避免 RBF 在小样本高维场景过拟合
- **在线 SVM (线性)**: `SGDClassifier(loss='hinge', penalty='l2')`, 通过 `partial_fit` 支持增量学习, 内存复杂度 O(d)
- **在线 SVM (RBF 核近似)**: `RBFSampler` (随机 Fourier 特征, Bochner 定理) → `StandardScaler` → `SGDClassifier`。使用 Monte Carlo 近似将 RBF 核映射为显式特征空间
- **类别不平衡处理**: `SGDClassifier(class_weight=dict)` 预先计算权重 (避免 `'balanced'` 不兼容 `partial_fit`); 批处理 SVC 使用 `class_weight='balanced'`; 负样本下采样至 max(5000, n_pos×5)

**阈值选择**: Youden 指数 (最大化 TPR-FPR)、EER (等错误率)、Quantile (正样本分位数)、Fixed (固定阈值)

**验证器并行化**: `ThreadPoolExecutor` 并行训练各用户验证器 (max 8 workers)

### 1D-CNN — 逐用户二分类验证

- **骨干网络**: 1×1 瓶颈降维 (200→64) → 4 层 1D 卷积 (64→128→256→512), BatchNorm + Dropout
- **Conv1d 沿子载波轴滑动**, 将窗口时间步作为独立通道处理
- **输出**: Linear(512, 1) → sigmoid 概率
- **损失函数**: `BCEWithLogitsLoss`
- **优化器**: AdamW (weight_decay=1e-4)
- **学习率调度**: CosineAnnealingLR (eta_min = lr × 1e-3)
- **早停策略**: 监控验证集 HTER
- **硬件优化**: AMP 混合精度、梯度累积、梯度检查点 (可配置), CUDA OOM 自动降级
- **类别不平衡处理**: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` 代价敏感惩罚 + 负样本下采样至 max(5000, n_pos×5)
- **推理**: verifier 按需 CPU→GPU 加载, 推理后释放显存; 支持 batch_size 配置防止大批量 OOM
- **配置项**: `use_checkpoint=True/False` 控制检查点 (省显存 vs 训练速度)

---

## 评估指标

### 主指标: HTER (Half Total Error Rate)

HTER = (FAR + FRR) / 2

### 分量指标

| 指标 | 公式 | 描述 |
|------|------|------|
| **FAR** | FP / (FP + TN) | 错误接受率 — impostor 被误判为 genuine |
| **FRR** | FN / (FN + TP) | 错误拒绝率 — genuine 被误判为 impostor |
| **准确率** | (TP + TN) / N | 全局分类准确率 |
| **F1 Score** | 2PR / (P + R) | 精确率与召回率的调和平均 |

### 每用户指标

每位注册用户独立计算 FAR、FRR、HTER 和阈值。系统同时输出跨用户的均值、标准差、最优/最差 HTER。

### 训练日志

所有训练结果以 JSONL 格式记录至 `logs/training_log.jsonl`，包含时间戳、模型类型、数据源、状态、耗时、配置参数和完整评估指标。

---

## 环境安装

### 环境要求

- **Python**: 3.10+
- **操作系统**: Windows / macOS / Linux
- **GPU** (可选): CUDA 兼容 GPU (CNN 训练推荐, ≥8GB 显存)

### 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖: `numpy`, `scipy`, `scikit-learn`, `torch>=2.0`, `streamlit`, `psutil`, `matplotlib`, `pandas`

---

## 运行方式

### Streamlit 工作台

```bash
streamlit run app_auth.py
```

工作台包含两大阶段:

1. **注册阶段 (训练)**:
   - **基础训练**: 完整参数配置 (窗口大小、特征组、PCA、降噪方法、在线SVM 选项)
   - **参数研究**: 遍历不同训练样本数, 生成样本量-性能曲线
   - **模型对比**: SVM vs CNN 认证性能对比
   - **🧪 实验**: 一键运行论文实验 E1~E4, 支持数据源兼容性自动过滤

2. **认证阶段 (推理)**:
   - **单次认证**: 上传样本 → 分数分布图 + 判决 (支持 RSSI 切片模式和 CSI 组合模式)
   - **持续认证**: 时序监控 + 平滑分数 + 决策条 + 状态切换检测

### 命令行

```bash
# SVM 认证
python -c "from scripts.pipeline_runner import run_npy_authentication_svm; print(run_npy_authentication_svm(seed=42))"

# 在线 SVM (RBF 核近似)
python -c "from scripts.pipeline_runner import run_npy_authentication_svm; print(run_npy_authentication_svm(seed=42, use_online_svm=True, online_kernel='rbf'))"

# CNN 认证
python -c "from scripts.pipeline_runner import run_npy_authentication_cnn; print(run_npy_authentication_cnn(seed=42, epochs=20, batch_size=64))"
```

---

## 配置参数

### 流水线配置 (`scripts/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `random_seed` | 42 | 随机种子 (可复现性) |
| `test_size` | 0.2 | 测试集比例 |
| `window_size` | 200 | 滑动窗口大小 (时间步) |
| `step_size` | 100 | 滑动步长 |
| `pca_variance` | 0.9019 | PCA 保留方差比例 |
| `use_pca` | True | 启用 PCA 降维 |
| `feature_groups` | `(spectral, statistical, temporal)` | 启用的特征组 |
| `csi_selected_actions` | `(5,31,32,34,35,36,37,39,40,47)` | CSI 精选动作, None=全部55个 |
| `csi_denoise` | `None` | CSI 原始信号降噪: `hampel`/`savgol`/`butterworth` |

### SVM 配置 (`scripts/models/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `svm_C` | 1.0 | SVC 正则化系数 |
| `svm_gamma` | `"scale"` | RBF 核 gamma (自适应维度) |
| `threshold_method` | `"youden"` | 阈值选择方法 |
| `cv_folds` | 3 | 交叉验证折数 |

### CNN 配置 (`scripts/models/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `conv_channels` | `(64, 128, 256, 512)` | 卷积通道数 (实验 E4 使用轻量 `(32, 64, 128, 192)`) |
| `kernel_size` | 5 | 卷积核大小 |
| `hidden_units` | 512 | 全连接隐藏单元 (实验 E4 使用 256) |
| `dropout_rates` | `(0.3, 0.5)` | Dropout 比率 |
| `epochs` | 20 | 训练轮数 (实验 E4 使用 10) |
| `batch_size` | 64 | 批次大小 (实验 E4 使用 96) |
| `learning_rate` | 1e-3 | 学习率 |
| `early_stop_patience` | 5 | 早停耐心轮数 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 (实验 E4 使用 1) |
| `use_checkpoint` | True | 梯度检查点 (实验 E4 关闭以提速 ~40%) |

---

## 性能优化

### 内存管理

- **LRU 数据缓存**: 避免重复文件 I/O, 配备命中率遥测
- **内存映射 I/O**: 大规模窗口数据使用 `np.memmap` 零拷贝加载, OS 按需分页
- **文件句柄管理**: `np.load(mmap_mode)` 后显式 `_mmap.close()` 防止 fd 泄漏; `build_windows_memmap` 写入后显式释放 memmap
- **float32 精度**: FFT 输出 complex64 而非 complex128, 在线 SVM transform 后 float64→float32
- **索引洗牌**: 替代 `sklearn.utils.shuffle` 避免全量数组拷贝 (OOM 修复)
- **GPU 推理**: CNN verifier 按需 CPU→GPU, 推理后释放 (`v.cpu()` + `empty_cache()`)
- **自动降级**: 系统内存 < 15% 或 GPU 显存 > 90% 时自动缩减批次与并行度

### 训练优化

#### CSI 流水线
- **并行特征提取**: `ThreadPoolExecutor` 8 workers 并行处理文件 I/O 与特征计算
- **Header-only 扫描**: `np.load(mmap_mode='r')` 仅映射 header 获取 shape, 不加载数据
- **并行验证器训练**: 逐用户 SVC/SGD 通过 `ThreadPoolExecutor` 并行

#### CNN 训练
- **负样本下采样**: 每用户负样本上限 max(5000, n_pos×5), 3-4× 加速
- **Cosine Annealing LR**: 平滑余弦衰减
- **HTER 早停**: 直接优化认证目标指标
- **混合精度训练 (AMP)**: 减少 30-50% 显存
- **梯度累积**: 在小 GPU 上模拟大批次
- **可配置检查点**: `use_checkpoint=False` 关闭以 ~40% 提速 (需充足显存)
- **OOM 自动恢复**: 显存溢出时自动减半批次并重建 DataLoader

#### PCA
- **自适应 SVD 求解器**: `svd_solver='auto'` 自动选择 randomized/full/arpack

### 流水线缓存

中间结果以参数哈希命名缓存于 `cache/`。相同参数重跑时跳过已完成阶段。Hash 包含 `csi_selected_actions` 和 `csi_denoise`, 参数变更自动触发重建。有效期 7 天, 总大小上限 50GB。

```bash
rm -rf cache/*   # 手动清空缓存
```

---

## API 参考

```python
from scripts import (
    # 配置
    PipelineConfig, WindowConfig, PreprocessConfig,
    SVMConfig, CNNConfig, CNNTrainConfig,

    # 数据加载
    load_rssi_data, load_csi_data, load_npy_matrix, parse_npy_filename,

    # 数据划分
    DatasetSplitter, Sample,

    # 窗口构建
    WindowBuilder, WindowProcessor,

    # 特征工程
    FeatureExtractor, FeatureProcessor,

    # 模型
    SVMAuthenticationTrainer, OnlineSVMVerifier,
    CNNTrainer, CNNInference,

    # 流水线
    AuthPipeline, run_authentication_pipeline,
    run_npy_authentication_svm, run_npy_authentication_cnn,

    # 实验模块
    from experiments import (
        ExperimentRunner,
        render_e1_results, render_e2_results,
        render_e3_results, render_e4_results,
    )
)
```

---

## 常见问题

### 推理时特征维度不匹配

模型训练时保存了完整的特征提取配置 (`feature_config`) 和 `feature_dim`。推理时自动读取并验证。错误信息会给出具体修复建议 (如调整窗口大小)。解决方案:

1. 使用与训练时相同的 `window_size` 构建滑动窗口
2. 或删除旧模型重新训练

### SVM 认证 FAR 偏高

1. 启用 PCA 降维 (`use_pca=True`) — 对高维 CSI 特征 (7,830 维) 至为关键
2. 尝试 `threshold_method='eer'` 均衡 FAR/FRR
3. CSI 数据尝试在线 SVM `kernel='rbf'` (RBFSampler 核近似)
4. 确认 `feature_groups` 包含 `temporal` (时域特征有助于捕获信号动态)

### CNN 训练显存不足

1. 减小 `batch_size` (64 → 32 或 16)
2. 增大 `gradient_accumulation_steps` (4 → 8)
3. 减小 `max_files_per_subject` (100 → 50)
4. 保持 `use_checkpoint=True` (默认) 以计算换显存
5. 使用 CPU 模式: `CUDA_VISIBLE_DEVICES=""`

### CSI 数据加载缓慢

- 启用异步加载可获 3-5× 加速: `load_csi_data(use_async=True, max_workers=8)`
- 清除过期缓存: `rm -rf cache/*`

### 流水线重复计算

中间文件以参数哈希命名，不同参数组合自动独立缓存。需清除特定参数缓存时:

```bash
rm -f data/rssi_*_{params_hash}*.pkl
```

### 认证结果全相同

检查是否命中旧版本缓存（不含参数哈希）。清除缓存后重试:

```bash
rm -rf cache/* data/rssi_processed*.pkl data/rssi_windowed*.pkl data/rssi_split*.pkl
```

### CSI 模型对比 SVM 速度慢

CSI 模型对比自动使用在线 SVM (SGD+RBFSampler)，比批处理 SVC 快 10-50×。基础训练可在高级选项中勾选"在线SVM"启用。

### 模型训练 OOM

在线 SVM: 已优化 float32 变换和索引洗牌。如仍 OOM，减少 `max_files_per_subject`。
CNN: 见上方"CNN 训练显存不足"。

---

## 许可证

内部项目 — 仅供学术研究与教育使用。
