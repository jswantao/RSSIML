# RSSI 身份识别与分类系统

## 1. 项目概述

本项目实现了 RSSI 场景下从数据预处理到模型训练再到可视化推理的完整流程，包含传统机器学习与 1D CNN 两条链路。

- 数据来源：20 个 MAT 文件（变量名为 RSSI）
- 单文件规模：约 20000 x 52
- 任务类型：
  - 身份分类（多分类）
  - 身份识别（开放集，包含 Unknown 拒识）

## 2. 处理与训练流水线

```text
raw/*.mat
  -> 数据集划分 (scripts/split_rssi_dataset.py)
  -> 滑窗构建 (scripts/build_sliding_windows.py)
  -> 频域+统计特征 / PCA / 归一化 (scripts/process_features_pca_norm.py)
  -> 传统模型训练 (scripts/train_and_validate_models.py)
  -> CNN 模型训练 (scripts/train_cnn_models.py)
  -> Web 可视化与推理 (main.py)
```

### 2.1 数据划分

- 分类任务：subject_stratified_file_split
  - 每个受试者在训练集和测试集都出现
  - 输出：data/rssi_split_classification.pkl
- 识别任务：open_set_known_unknown_split
  - 先划分 known/unknown 人员
  - known 人员再按文件划分 train/test
  - unknown 人员全部进入测试集
  - 输出：data/rssi_split_identification.pkl

### 2.2 滑窗构建

- 默认窗口大小：200
- 默认步长：100
- 输出：
  - data/rssi_windowed_classification.pkl
  - data/rssi_windowed_identification.pkl

### 2.3 特征工程

- 频域特征：窗口去均值后 FFT 低频幅值
- 统计特征：标准差、偏度、峰度、过零率
- 可选 PCA：默认保留方差 0.9019
- 归一化：MinMax 映射到 [-1, 1]
- 输出：
  - data/rssi_processed_classification.pkl
  - data/rssi_processed_identification.pkl

### 2.4 传统模型

- 分类：SVM、RandomForest、XGBoost、LightGBM
- 识别：基于类别中心与距离阈值的开放集识别
- 指标：Accuracy、Precision、Recall、F1（分类含 CV 指标）
- 输出：
  - models/best_classification_model.joblib
  - models/identification_model.joblib
  - results/classification_metrics.json
  - results/identification_metrics.json

### 2.5 1D CNN 模型

- 结构：Conv1d + BN + ReLU + Pool + Dropout + FC
- 训练：支持验证集、早停、随机种子控制
- 输出：
  - models/cnn_classification.pt
  - models/cnn_identification.pt
  - results/cnn_classification_metrics.json
  - results/cnn_identification_metrics.json

## 3. 项目结构

```text
RSSIML/
├─ raw/                                # 原始 MAT 数据（已上传）
├─ data/                               # 中间数据（默认不提交）
├─ models/                             # 模型文件（默认不提交）
├─ results/                            # 评估结果（默认不提交）
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
│  ├─ test_feature_mock.py
│  └─ test_cnn_mock.py
├─ main.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 4. 环境安装

建议 Python 3.10 及以上。

```bash
pip install -r requirements.txt
```

## 5. 运行方式

### 5.1 命令行：生成划分

```bash
python -m scripts.split_rssi_dataset --test-size 0.2 --seed 42 --unknown-ratio 0.4
```

### 5.2 命令行：仅数据流水线

```bash
python -c "from scripts.pipeline_runner import run_classification_data_pipeline; print(run_classification_data_pipeline())"
python -c "from scripts.pipeline_runner import run_identification_data_pipeline; print(run_identification_data_pipeline())"
```

### 5.3 命令行：传统模型训练

```bash
python -m scripts.train_and_validate_models --task classification --cv-folds 5 --seed 42
python -m scripts.train_and_validate_models --task identification --threshold-quantile 0.95 --seed 42
```

### 5.4 命令行：CNN 训练

```bash
python -m scripts.train_cnn_models --task classification --epochs 20 --batch-size 64 --learning-rate 0.001 --val-ratio 0.2 --early-stop-patience 5 --seed 42
python -m scripts.train_cnn_models --task identification --epochs 20 --batch-size 64 --learning-rate 0.001 --val-ratio 0.2 --early-stop-patience 5 --seed 42
```

### 5.5 图形化运行

```bash
streamlit run main.py
```

页面支持：

- 数据流水线执行（分类 + 识别）
- 传统模型训练与评估
- CNN 训练与推理
- 单文件推理与可视化

## 6. Git 提交策略说明

- 已提交：源码、README、raw 原始数据
- 默认忽略：data、models、results、raw.zip 及常见模型缓存文件
- 如需提交训练产物，请按需修改 .gitignore

## 7. 常见问题

- 若出现模型与数据维度不匹配，请先重新执行对应任务的数据流水线，再训练模型。
- 若 VS Code 报第三方库导入错误（如 streamlit、seaborn、torch），通常是当前解释器环境未安装依赖，请重新执行 pip install -r requirements.txt 并切换到正确解释器。
