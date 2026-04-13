# -*- coding: utf-8 -*-
"""RSSI 身份识别系统主入口模块。

提供基于 Streamlit 的交互式界面，整合数据处理、模型训练、推理和可视化功能。
支持传统机器学习（SVM/RF/LR）和深度学习（1D-CNN）两种技术路线。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

# 本地模块导入
from scripts.build_sliding_windows import WindowBuilder, WindowConfig
from scripts.config import PipelineConfig
from scripts.pipeline_runner import (
    run_data_pipeline,
    run_classification_pipeline,
    run_identification_pipeline,
)
from scripts.train_cnn_models import (
    CNNTrainer,
    TrainingConfig as CNNTrainingConfig,
    train_cnn_classification,
    train_cnn_identification,
)

# =============================================================================
# 全局配置与初始化
# =============================================================================

_CONFIG = PipelineConfig.from_root()

# 页面配置
st.set_page_config(
    page_title="RSSI 身份识别系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 以美化界面
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .status-card {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 5px solid #ccc;
        margin-bottom: 5px;
    }
    .status-ready {
        border-left-color: #4CAF50;
    }
    .status-missing {
        border-left-color: #F44336;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">RSSI 身份识别与分类系统</div>', unsafe_allow_html=True)


# =============================================================================
# 工具函数
# =============================================================================

@st.cache_resource(show_spinner=False)
def _get_font_prop() -> fm.FontProperties:
    """获取中文字体属性，用于 Matplotlib 图表。

    Returns:
        FontProperties 对象。
    """
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/PingFang.ttc",              # macOS
        "C:/Windows/Fonts/msyh.ttc",                       # Windows 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",                     # Windows 宋体
    ]
    for path in font_paths:
        if Path(path).exists():
            return fm.FontProperties(fname=path)
    return fm.FontProperties()  #  fallback


_FONT_PROP = _get_font_prop()


def _check_file_status(file_path: Path) -> bool:
    """检查文件是否存在。

    Args:
        file_path: 文件路径。

    Returns:
        True 如果存在，否则 False。
    """
    return file_path.exists()


def _render_status_card(name: str, exists: bool, icon: str = "📄") -> None:
    """渲染状态卡片。

    Args:
        name: 组件名称。
        exists: 是否存在。
        icon: 图标。
    """
    css_class = "status-ready" if exists else "status-missing"
    status_text = "已就绪" if exists else "缺失"
    st.markdown(
        f"""
        <div class="status-card {css_class}">
            <strong>{icon} {name}</strong>: {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_time_series(data: np.ndarray, title: str, x_label: str, y_label: str) -> None:
    """绘制时间序列数据。

    Args:
        data: 二维数组 (Time, Channels)。
        title: 图表标题。
        x_label: X 轴标签。
        y_label: Y 轴标签。
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    time_steps = np.arange(data.shape[0])
    for i in range(min(data.shape[1], 5)):  # 最多绘制前5个通道
        ax.plot(time_steps, data[:, i], label=f"Ch {i+1}", alpha=0.7)
    
    ax.set_title(title, fontproperties=_FONT_PROP)
    ax.set_xlabel(x_label, fontproperties=_FONT_PROP)
    ax.set_ylabel(y_label, fontproperties=_FONT_PROP)
    ax.legend(prop=_FONT_PROP)
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# 训练页面逻辑
# =============================================================================

def _render_training_page() -> None:
    """渲染训练页面。"""
    st.header("模型训练工作台")

    # --- 1. 系统状态检查 ---
    with st.expander("系统资源状态检查", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # 数据文件状态
        with col1:
            st.subheader("数据文件")
            _render_status_card("分类数据", _check_file_status(_CONFIG.data_dir / "rssi_processed_classification.pkl"))
            _render_status_card("识别数据", _check_file_status(_CONFIG.data_dir / "rssi_processed_identification.pkl"))
            _render_status_card("CNN分类窗口", _check_file_status(_CONFIG.data_dir / "rssi_windowed_classification.pkl"))
            _render_status_card("CNN识别窗口", _check_file_status(_CONFIG.data_dir / "rssi_windowed_identification.pkl"))

        # 传统模型状态
        with col2:
            st.subheader("传统模型")
            _render_status_card("分类模型", _check_file_status(_CONFIG.model_dir / "best_RandomForest_classification.joblib"))
            _render_status_card("识别模型", _check_file_status(_CONFIG.model_dir / "identification_model.pkl"))

        # CNN 模型状态
        with col3:
            st.subheader("CNN 模型")
            _render_status_card("CNN分类模型", _check_file_status(_CONFIG.model_dir / "cnn_best_model.pth")) # 注意：CNNTrainer 保存为 pth
            _render_status_card("CNN识别模型", _check_file_status(_CONFIG.model_dir / "cnn_best_model.pth")) # 注意：通常共用一个文件名或需区分，这里假设最新训练覆盖

    st.markdown("---")

    # --- 2. 全局参数设置 ---
    st.subheader("全局参数设置")
    col_params1, col_params2 = st.columns(2)
    
    with col_params1:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, step=0.05, help="用于数据划分")
        seed = st.number_input("随机种子", value=42, min_value=0, help="保证结果可复现")
        window_size = st.number_input("滑动窗口大小", value=200, min_value=10, help="时间步长")
        step_size = st.number_input("滑动步长", value=100, min_value=1, help="窗口移动步长")

    with col_params2:
        pca_variance = st.slider("PCA 方差保留比例", 0.5, 1.0, 0.9019, step=0.01, help="仅对传统模型有效")
        use_pca = st.checkbox("启用 PCA 降维", value=True, help="仅对传统模型有效")
        
        st.markdown("CNN 参数")
        cnn_epochs = st.number_input("CNN Epochs", min_value=1, max_value=200, value=20, step=1)
        cnn_batch_size = st.number_input("CNN Batch Size", min_value=8, max_value=512, value=64, step=8)
        cnn_lr = st.number_input("CNN Learning Rate", min_value=0.00001, max_value=0.1, value=0.001, step=0.0005, format="%.5f")

    # --- 3. 数据预处理板块 ---
    st.subheader("步骤 1: 数据预处理")
    st.caption("生成滑动窗口、提取特征（传统模型）或准备窗口数据（CNN）。")
    
    col_data1, col_data2 = st.columns(2)
    
    with col_data1:
        if st.button("生成分类任务数据", key="btn_gen_class_data"):
            with st.spinner("正在生成分类任务数据..."):
                try:
                    result = run_data_pipeline(
                        test_size=test_size,
                        seed=seed,
                        window_size=window_size,
                        step_size=step_size,
                        pca_variance=pca_variance,
                        use_pca=use_pca,
                        task="classification",
                    )
                    st.success("分类数据生成成功！")
                    st.json(result.get("process_meta", {}))
                except Exception as e:
                    st.error(f"生成失败: {str(e)}")

    with col_data2:
        if st.button("生成识别任务数据", key="btn_gen_id_data"):
            with st.spinner("正在生成识别任务数据..."):
                try:
                    result = run_data_pipeline(
                        test_size=test_size,
                        seed=seed,
                        window_size=window_size,
                        step_size=step_size,
                        pca_variance=pca_variance,
                        use_pca=use_pca,
                        task="identification",
                    )
                    st.success("识别数据生成成功！")
                    st.json(result.get("process_meta", {}))
                except Exception as e:
                    st.error(f"生成失败: {str(e)}")

    st.markdown("---")

    # --- 4. 传统模型训练板块 ---
    st.subheader("步骤 2: 传统机器学习模型训练")
    st.caption("使用 SVM, Random Forest, Logistic Regression 进行训练。需要先生成『处理后数据』。")
    
    col_trad1, col_trad2 = st.columns(2)
    
    with col_trad1:
        if st.button("训练传统分类模型", key="btn_train_trad_class"):
            with st.spinner("正在训练传统分类模型..."):
                try:
                    result = run_classification_pipeline(
                        test_size=test_size,
                        seed=seed,
                        window_size=window_size,
                        step_size=step_size,
                        pca_variance=pca_variance,
                        use_pca=use_pca,
                    )
                    st.success("传统分类模型训练完成！")
                    st.json(result.get("model_metrics", {}))
                except Exception as e:
                    st.error(f"训练失败: {str(e)}")

    with col_trad2:
        if st.button("训练传统识别模型", key="btn_train_trad_id"):
            with st.spinner("正在训练传统识别模型..."):
                try:
                    result = run_identification_pipeline(
                        test_size=test_size,
                        seed=seed,
                        window_size=window_size,
                        step_size=step_size,
                        pca_variance=pca_variance,
                        use_pca=use_pca,
                    )
                    st.success("传统识别模型训练完成！")
                    st.json(result.get("model_metrics", {}))
                except Exception as e:
                    st.error(f"训练失败: {str(e)}")

    st.markdown("---")

    # --- 5. CNN 模型训练板块 ---
    st.subheader("步骤 3: 1D-CNN 深度学习模型训练")
    st.caption("使用卷积神经网络直接处理滑动窗口数据。需要先生成『窗口数据』。")
    
    col_cnn1, col_cnn2 = st.columns(2)
    
    with col_cnn1:
        if st.button("训练 CNN 分类模型", key="btn_train_cnn_class"):
            with st.spinner("正在训练 CNN 分类模型..."):
                try:
                    # CNN 训练直接使用窗口数据文件
                    data_file = _CONFIG.data_dir / "rssi_windowed_classification.pkl"
                    if not data_file.exists():
                        st.warning("未找到窗口数据文件，请先执行『生成分类任务数据』。")
                    else:
                        result = train_cnn_classification(
                            data_file=data_file,
                            epochs=cnn_epochs,
                            batch_size=cnn_batch_size,
                            learning_rate=cnn_lr,
                            val_ratio=0.2, # 固定验证集比例
                            early_stop_patience=5,
                            seed=seed,
                        )
                        st.success("CNN 分类模型训练完成！")
                        st.json(result.get("classification", {}))
                except Exception as e:
                    st.error(f"训练失败: {str(e)}")

    with col_cnn2:
        if st.button("训练 CNN 识别模型", key="btn_train_cnn_id"):
            with st.spinner("正在训练 CNN 识别模型..."):
                try:
                    # CNN 训练直接使用窗口数据文件
                    data_file = _CONFIG.data_dir / "rssi_windowed_identification.pkl"
                    if not data_file.exists():
                        st.warning("未找到窗口数据文件，请先执行『生成识别任务数据』。")
                    else:
                        result = train_cnn_identification(
                            data_file=data_file,
                            epochs=cnn_epochs,
                            batch_size=cnn_batch_size,
                            learning_rate=cnn_lr,
                            val_ratio=0.2,
                            early_stop_patience=5,
                            seed=seed,
                        )
                        st.success("CNN 识别模型训练完成！")
                        st.json(result.get("identification", {}))
                except Exception as e:
                    st.error(f"训练失败: {str(e)}")


# =============================================================================
# 推理页面逻辑
# =============================================================================

def _render_inference_page() -> None:
    """渲染推理页面。"""
    st.header("模型推理工作台")

    # --- 1. 模型选择 ---
    model_type = st.selectbox(
        "选择推理模型类型",
        [
            "传统-分类模型 (Traditional Classification)",
            "传统-识别模型 (Traditional Identification)",
            "CNN-分类模型 (CNN Classification)",
            "CNN-识别模型 (CNN Identification)",
        ],
    )

    # --- 2. 文件上传 ---
    uploaded_file = st.file_uploader("上传待推理的 MAT 文件 (wipin*.mat)", type=["mat"])

    if uploaded_file is not None:
        try:
            # 加载原始数据
            mat_data = loadmat(uploaded_file)
            if "RSSI" not in mat_data:
                st.error("上传的 MAT 文件中未找到 'RSSI' 键。请检查文件格式。")
                return
            
            rssi_matrix = mat_data["RSSI"].astype(np.float32)
            
            # 展示原始数据
            st.subheader("原始 RSSI 信号预览")
            _plot_time_series(rssi_matrix, "原始 RSSI 信号", "时间点", "RSSI (dBm)")

            # 滑动窗口处理
            window_config = WindowConfig(window_size=200, step_size=100)
            window_builder = WindowBuilder(window_config)
            windows = window_builder.build(rssi_matrix)

            if windows.shape[0] == 0:
                st.warning("数据长度不足，无法生成滑动窗口。")
                return

            st.write(f"生成了 {windows.shape[0]} 个滑动窗口，形状: {windows.shape}")

            # --- 3. 根据模型类型执行推理 ---
            st.subheader("推理结果")
            
            if "传统" in model_type:
                _perform_traditional_inference(model_type, windows, rssi_matrix)
            elif "CNN" in model_type:
                _perform_cnn_inference(model_type, windows, rssi_matrix)

        except Exception as e:
            st.error(f"推理过程中发生错误: {str(e)}")
            st.exception(e)


def _perform_traditional_inference(model_type: str, windows: np.ndarray, raw_data: np.ndarray) -> None:
    """执行传统模型推理。

    Args:
        model_type: 模型类型字符串。
        windows: 滑动窗口数据。
        raw_data: 原始 RSSI 数据。
    """
    # 确定模型路径和预处理文件路径
    if "分类" in model_type:
        model_path = _CONFIG.model_dir / "best_RandomForest_classification.joblib"
        proc_file_path = _CONFIG.data_dir / "rssi_processed_classification.pkl"
    else:
        model_path = _CONFIG.model_dir / "identification_model.pkl"
        proc_file_path = _CONFIG.data_dir / "rssi_processed_identification.pkl"

    # 检查文件存在性
    if not model_path.exists():
        st.error(f"模型文件不存在: {model_path.name}。请先训练模型。")
        return
    
    if not proc_file_path.exists():
        st.error(f"预处理配置文件不存在: {proc_file_path.name}。无法进行特征提取。")
        return

    # 加载模型
    model = joblib.load(model_path)
    
    # 加载预处理组件 (PCA, Scaler)
    with open(proc_file_path, 'rb') as f:
        proc_data = pickle.load(f)

    # 兼容两套键名：新流程使用 pca_model/scaler_model，旧流程可能使用 pca/scaler
    pca = proc_data.get('pca_model') or proc_data.get('pca')
    scaler = proc_data.get('scaler_model') or proc_data.get('scaler')

    # 特征提取
    from scripts.process_features_pca_norm import FeatureExtractor
    feat_extractor = FeatureExtractor()
    
    # 注意：FeatureExtractor 期望输入是 (N, W, C)
    features = feat_extractor.extract_features(windows)

    # 应用 PCA
    if pca:
        features = pca.transform(features)
    
    # 应用 Scaler
    if scaler:
        features = scaler.transform(features)

    if "识别" in model_type:
        if not hasattr(model, "classifier"):
            st.error("识别模型文件格式不正确，缺少 classifier 字段。")
            return

        classifier = cast(Any, model.classifier)
        centroids = np.asarray(getattr(model, "centroids", None))
        threshold = float(getattr(model, "threshold", 0.0))
        label_encoder = getattr(model, "label_encoder", None)

        if centroids.size == 0 or label_encoder is None:
            st.error("识别模型文件不完整，缺少 centroids 或 label_encoder。")
            return

        expected_dim = getattr(classifier, "n_features_in_", None)
        if expected_dim is not None and int(features.shape[1]) != int(expected_dim):
            st.error(
                "模型输入维度不匹配："
                f"当前推理特征为 {features.shape[1]} 维，"
                f"模型期望 {int(expected_dim)} 维。"
                "请重新生成数据并重新训练当前任务模型。"
            )
            return

        y_pred_enc = classifier.predict(features)
        predicted_centroids = centroids[y_pred_enc]
        distances = np.linalg.norm(features - predicted_centroids, axis=1)
        predictions = np.where(distances <= threshold, "known", "unknown")
    else:
        expected_dim = getattr(model, "n_features_in_", None)
        if expected_dim is not None and int(features.shape[1]) != int(expected_dim):
            st.error(
                "模型输入维度不匹配："
                f"当前推理特征为 {features.shape[1]} 维，"
                f"模型期望 {int(expected_dim)} 维。"
                "请重新生成数据并重新训练当前任务模型。"
            )
            return

        predictions = model.predict(features)
    
    # 统计投票结果
    unique, counts = np.unique(predictions, return_counts=True)
    vote_df = pd.DataFrame({'Identity': unique, 'Count': counts})
    vote_df['Percentage'] = vote_df['Count'] / vote_df['Count'].sum()
    vote_df = vote_df.sort_values(by='Count', ascending=False)

    top_identity = vote_df.iloc[0]['Identity']
    top_percentage = vote_df.iloc[0]['Percentage']

    # 展示结果
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="🏆 最高票数身份", value=str(top_identity), delta=f"{top_percentage:.1%}")
    with col2:
        st.dataframe(vote_df.style.format({"Percentage": "{:.2%}"}), use_container_width=True)

    # 如果是识别模型，额外显示 Known/Unknown 分布
    if "识别" in model_type:
        known_count = (predictions == "known").sum()
        unknown_count = (predictions == "unknown").sum()
        st.bar_chart({"Known": [known_count], "Unknown": [unknown_count]})


def _perform_cnn_inference(model_type: str, windows: np.ndarray, raw_data: np.ndarray) -> None:
    """执行 CNN 模型推理。

    Args:
        model_type: 模型类型字符串。
        windows: 滑动窗口数据。
        raw_data: 原始 RSSI 数据。
    """
    if "分类" in model_type:
        candidate_paths = [
            _CONFIG.model_dir / "cnn_classification.pt",
            _CONFIG.model_dir / "cnn_best_model.pth",
        ]
    else:
        candidate_paths = [
            _CONFIG.model_dir / "cnn_identification.pt",
            _CONFIG.model_dir / "cnn_best_model.pth",
        ]

    model_path = next((path for path in candidate_paths if path.exists()), None)
    
    if model_path is None:
        st.error("CNN 模型文件不存在，请先训练对应的 CNN 模型。")
        return

    try:
        from scripts.train_cnn_models import CNNInference
        cnn_inference = CNNInference(model_path)
        encoder = cnn_inference.encoder
        distances = np.array([], dtype=np.float32)
        probs_np = np.empty((0, len(encoder.classes_)), dtype=np.float32)

        if "识别" in model_type:
            predictions, distances = cnn_inference.predict_open_set(windows)
        else:
            predictions = cnn_inference.predict(windows)
            import torch

            x_tensor = torch.tensor(windows, dtype=torch.float32).transpose(1, 2)
            with torch.no_grad():
                logits = cnn_inference.model(x_tensor)
                probs_np = torch.softmax(logits, dim=1).cpu().numpy()

        # 统计投票
        unique, counts = np.unique(predictions, return_counts=True)
        vote_df = pd.DataFrame({'Identity': unique, 'Count': counts})
        vote_df['Percentage'] = vote_df['Count'] / vote_df['Count'].sum()
        vote_df = vote_df.sort_values(by='Count', ascending=False)

        top_identity = vote_df.iloc[0]['Identity']
        top_percentage = vote_df.iloc[0]['Percentage']

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="最高票数身份", value=str(top_identity), delta=f"{top_percentage:.1%}")
        with col2:
            st.dataframe(vote_df.style.format({"Percentage": "{:.2%}"}), use_container_width=True)

        if "识别" in model_type:
            st.subheader("前 5 个窗口的开集距离")
            distance_table = pd.DataFrame({"distance": distances[:5]})
            st.dataframe(distance_table.style.format("{:.4f}"), use_container_width=True)
        else:
            # 显示前几个窗口的概率分布
            st.subheader("前 5 个窗口的预测概率")
            probability_table = pd.DataFrame(probs_np[:5], columns=encoder.classes_)
            st.dataframe(probability_table.style.format("{:.4f}"), use_container_width=True)

    except Exception as e:
        st.error(f"CNN 推理失败: {str(e)}")
        st.exception(e)


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口函数。"""
    # 侧边栏导航
    page = st.sidebar.radio(
        "页面导航",
        ["模型训练", "在线推理"],
        index=0,
    )

    if page == "模型训练":
        _render_training_page()
    elif page == "在线推理":
        _render_inference_page()


if __name__ == "__main__":
    main()