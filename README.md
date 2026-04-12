# RSSI 身份识别与分类系统

## 1. 项目概述

本项目实现了 RSSI 场景下的完整机器学习流水线，包含传统机器学习与 1D CNN 两类模型。

- 数据来源: 20 个 `.mat` 文件，变量名为 `RSSI`
- 单文件规模: `20000 x 52`
- 总规模: 1,040,000 个时间采样点
- 任务类型:
  - 身份分类: 5 人多分类
  - 身份识别: 未见人员检测（person-disjoint 开放集识别）

## 2. 流水线步骤

```text
原始数据(raw/*.mat)
    -> [1] 数据集划分 split_rssi_dataset.py
    -> [2] 滑窗构建 build_sliding_windows.py
    -> [3] 频域特征 + PCA + 归一化 process_features_pca_norm.py
    -> [4] 传统模型训练 train_and_validate_models.py
    -> [5] 1D CNN 训练 train_cnn_models.py
    -> [5] GUI 可视化 main.py (Streamlit)
```

### [1] 数据集划分

- 分类任务划分: `subject_stratified_file_split`
  - 每个人员在训练集和测试集中都出现
  - 输出: `data/rssi_split_classification.pkl`
- 识别任务划分: `person_disjoint_split`
  - 训练集和测试集人员集合完全不重叠
  - 测试集只包含训练集中未见过的人员
  - 输出: `data/rssi_split_identification.pkl`
- 兼容文件: `data/rssi_split.pkl`（默认指向分类任务划分）

### [2] 滑窗构建

- 窗口大小: 200
- 步长: 100
- 单文件样本数: 199 个窗口
- 输出:
  - 分类: `data/rssi_windowed_classification.pkl`
  - 识别: `data/rssi_windowed_identification.pkl`
  - 兼容: `data/rssi_windowed.pkl`

### [3] 频域特征 + PCA + 归一化

- 特征方式: 200 点窗口的低频 FFT 幅值 + 窗口统计特征（标准差、偏度、峰度、过零率）
- PCA 开关: 支持启用/关闭
- PCA 拟合策略: 仅在训练集上拟合，再对测试集 transform
- PCA: 默认保留 90.19% 方差
- 归一化: 映射到 `[-1, 1]`
- 输出:
  - 分类: `data/rssi_processed_classification.pkl`
  - 识别: `data/rssi_processed_identification.pkl`
  - 兼容: `data/rssi_processed.pkl`

### [4] 传统模型训练与评估

- 分类模型: SVM、RandomForest，若环境安装则自动启用 XGBoost / LightGBM
- 识别模型: One-Class SVM（未见人员检测）
- 指标: Accuracy、Precision、Recall、F1
- 输出:
  - `models/best_classification_model.joblib`
  - `models/identification_model.joblib`
  - `results/classification_metrics.json`
  - `results/identification_metrics.json`

### [5] 1D CNN 训练与评估

- 网络: Conv1d + BN + ReLU + Pool + Dropout
- 训练策略:
  - train/val/test 三段式
  - 按验证集指标选最优权重
  - 支持早停
- 分类输出:
  - `models/cnn_classification.pt`
  - `results/cnn_classification_metrics.json`
- 识别输出:
  - `models/cnn_identification.pt`
  - `results/cnn_identification_metrics.json`

## 3. 项目结构

```text
RSSIML/
├─ raw/                                # 原始数据
├─ data/                               # 中间数据
├─ models/                             # 模型文件
├─ results/                            # 评估结果
├─ scripts/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ split_rssi_dataset.py
│  ├─ build_sliding_windows.py
│  ├─ process_features_pca_norm.py
│  ├─ train_and_validate_models.py
│  ├─ train_cnn_models.py
│  └─ pipeline_runner.py
├─ tests/
│  ├─ test_pipeline_mock.py            # 传统流程 mock 测试
│  └─ test_cnn_mock.py                 # CNN mock 测试
├─ main.py                             # Streamlit GUI
├─ requirements.txt
└─ README.md
```

## 4. 环境安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 5. 命令行运行

### 5.1 生成两套划分

```bash
python -m scripts.split_rssi_dataset --test-size 0.2 --seed 42
```

### 5.2 分类任务（传统模型）

```bash
python -c "from scripts.pipeline_runner import runClassificationDataPipeline; runClassificationDataPipeline(testSize=0.2, seed=42, windowSize=200, stepSize=100, pcaVariance=0.9019, usePca=True)"
python -m scripts.train_and_validate_models --task classification
```

### 5.3 识别任务（传统模型，person-disjoint）

```bash
python -c "from scripts.pipeline_runner import runIdentificationDataPipeline; runIdentificationDataPipeline(testSize=0.2, seed=42, windowSize=200, stepSize=100, pcaVariance=0.9019, usePca=True)"
python -m scripts.train_and_validate_models --task identification
```

### 5.4 1D CNN 训练

```bash
python -m scripts.train_cnn_models --task classification --epochs 20 --batch-size 64 --val-ratio 0.2 --early-stop-patience 5
python -m scripts.train_cnn_models --task identification --epochs 20 --batch-size 64 --val-ratio 0.2 --early-stop-patience 5
```

### 5.5 一键图形化运行

```bash
streamlit run main.py
```

进入页面后可在训练页执行:

- 数据流水线（分类 + 识别）
- 传统分类/识别训练
- 1D CNN 分类/识别训练

## 6. 说明

- 识别任务采用未见人员检测语义，指标中的 `positive_subject` 为 `unseen_person`。
- 若出现模型与数据维度不匹配，请在训练页按当前参数重新执行对应任务的数据流水线并重训。
- Pylance 若提示第三方库导入问题，请确认 VS Code 选中的 Python 解释器与安装依赖的环境一致。
