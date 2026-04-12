from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.io import loadmat
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from scripts.build_sliding_windows import WindowBuilder, WindowConfig
from scripts.config import PipelineConfig
from scripts.pipeline_runner import run_classification_data_pipeline, run_identification_data_pipeline
from scripts.process_features_pca_norm import FeatureConfig, FeatureExtractor
from scripts.train_cnn_models import (
    CNNInference,
    CNNTrainer,
    TrainingConfig as CNNTrainingConfig,
)
from scripts.train_and_validate_models import TrainingPipeline as ModelTrainingPipeline

# 设置页面配置
st.set_page_config(page_title="RSSI 身份识别系统", page_icon="📶", layout="wide")

config = PipelineConfig.from_root()


def setup_matplotlib_font() -> None:
    """设置matplotlib中文字体回退。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_matplotlib_font()


def load_processed_payload(file_name: str = "rssi_processed_classification.pkl") -> dict[str, Any] | None:
    """加载处理后的数据负载。"""
    processed_path = config.data_dir / file_name
    if not processed_path.exists():
        return None
    with processed_path.open("rb") as f:
        return pickle.load(f)


def load_json_metrics(file_name: str) -> dict[str, Any] | None:
    """加载JSON格式的评估指标。"""
    result_path = config.result_dir / file_name
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_rssi_from_mat_object(mat_obj: dict[str, Any]) -> np.ndarray:
    """从MAT对象中提取RSSI数据。"""
    if "RSSI" not in mat_obj:
        keys = [k for k in mat_obj.keys() if not k.startswith("__")]
        raise ValueError(f"MAT 文件缺少 RSSI 键，实际键: {keys}")
    matrix = np.asarray(mat_obj["RSSI"], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"RSSI 数据维度错误: {matrix.shape}")
    return matrix


def build_inference_feature(
    matrix: np.ndarray,
    window_size: int,
    step_size: int,
    low_freq_bins: int,
    pca: Any,
    scaler: Any,
    use_pca: bool,
) -> np.ndarray:
    """构建推理用特征。"""
    windows = WindowBuilder(
        WindowConfig(window_size=window_size, step_size=step_size)
    ).build(matrix)
    if windows.shape[0] == 0:
        raise ValueError("输入序列长度小于窗口大小，无法生成滑窗样本")
    features = FeatureExtractor(
        FeatureConfig(low_freq_bins=low_freq_bins, use_pca=False)
    ).extract_features(windows)
    transformed = pca.transform(features) if use_pca else features
    normalized = scaler.transform(transformed).astype(np.float32)
    return normalized


def build_cnn_inference_windows(
    matrix: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """构建 1D CNN 的推理窗口。"""
    windows = WindowBuilder(
        WindowConfig(window_size=window_size, step_size=step_size)
    ).build(matrix)
    if windows.shape[0] == 0:
        raise ValueError("输入序列长度小于窗口大小，无法生成滑窗样本")
    return windows.astype(np.float32, copy=False)


def get_expected_feature_dim(model: Any) -> int | None:
    """获取模型期望输入特征维度。"""
    value = getattr(model, "n_features_in_", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def validate_data_model_compatibility(task_type: str, model_path: Path, data_payload: dict[str, Any]) -> tuple[bool, str]:
    """验证数据和模型的兼容性。
    
    Args:
        task_type: 任务类型 ("classification" 或 "identification")
        model_path: 模型文件路径
        data_payload: 数据负载
        
    Returns:
        (是否兼容, 错误信息)
    """
    if not model_path.exists():
        return False, "模型文件不存在"
    
    # 加载模型（识别任务可能是包含元数据的字典）
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict) and task_type == "identification" and "centroids" in loaded:
        x_test = np.asarray(data_payload["x_test"], dtype=np.float32)
        centroids = np.asarray(loaded["centroids"], dtype=np.float32)
        if centroids.ndim != 2:
            return False, "识别模型中心向量格式异常"
        if x_test.shape[1] != centroids.shape[1]:
            return False, f"模型与数据维度不匹配: 模型期望 {centroids.shape[1]}，当前为 {x_test.shape[1]}"
        return True, ""

    model = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
    
    # 获取数据维度
    x_test = np.asarray(data_payload["x_test"], dtype=np.float32)
    expected_dim = get_expected_feature_dim(model)
    
    if expected_dim is not None and x_test.shape[1] != expected_dim:
        return False, f"模型与数据维度不匹配: 模型期望 {expected_dim}，当前为 {x_test.shape[1]}"
    
    return True, ""


def render_confusion_and_importance() -> None:
    """渲染混淆矩阵和特征重要性图。"""
    payload = load_processed_payload("rssi_processed_classification.pkl")
    if payload is None:
        st.warning("未检测到处理后数据，请先在训练页执行完整流水线")
        return

    model_path = config.model_dir / "best_classification_model.joblib"
    if not model_path.exists():
        st.warning("未检测到分类模型，请先训练")
        return

    model = joblib.load(model_path)
    x_test = np.asarray(payload["x_test"], dtype=np.float32)
    y_test = np.asarray(payload["y_test"], dtype=object)

    encoder = LabelEncoder()
    encoder.fit(np.asarray(payload["y_train"], dtype=object))
    y_test_encoded = encoder.transform(y_test)

    expected_dim = get_expected_feature_dim(model)
    if expected_dim is not None and x_test.shape[1] != expected_dim:
        st.error(
            f"分类模型与当前处理数据维度不一致: 模型期望 {expected_dim}，当前为 {x_test.shape[1]}。"
        )
        st.info("请在训练页使用与当前数据处理一致的参数重新训练身份分类模型。")
        return

    pred_encoded = np.asarray(model.predict(x_test))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test_encoded, pred_encoded)
    class_labels = [str(v) for v in np.asarray(encoder.classes_, dtype=object).tolist()]
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax_cm,
    )
    ax_cm.set_xlabel("预测标签")
    ax_cm.set_ylabel("真实标签")
    ax_cm.set_title("分类任务混淆矩阵")
    st.pyplot(fig_cm)

    # 计算特征重要性
    feature_names = [f"PC{i + 1}" for i in range(x_test.shape[1])]
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(getattr(model, "feature_importances_"), dtype=np.float32)
    else:
        importance_result = permutation_importance(
            model,
            x_test,
            y_test_encoded,
            n_repeats=5,
            random_state=42,
            scoring="accuracy",
        )
        importance = np.asarray(getattr(importance_result, "importances_mean"), dtype=np.float32)

    # 绘制特征重要性图
    fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
    order = np.argsort(importance)[::-1]
    imp_df = pd.DataFrame(
        {
            "feature": np.asarray(feature_names)[order],
            "importance": importance[order],
        }
    )
    sns.barplot(
        data=imp_df,
        x="importance",
        y="feature",
        hue="feature",
        dodge=False,
        legend=False,
        palette="crest",
        ax=ax_imp,
    )
    ax_imp.set_xlabel("重要性")
    ax_imp.set_ylabel("特征")
    ax_imp.set_title("特征重要性")
    st.pyplot(fig_imp)


def _build_cnn_trainer(epochs: int, batch_size: int, learning_rate: float, val_ratio: float, early_stop_patience: int, seed: int) -> CNNTrainer:
    """构建 CNN 训练器。"""
    train_config = CNNTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        early_stop_patience=early_stop_patience,
        random_seed=seed,
    )
    return CNNTrainer(train_config=train_config)


def check_data_pipeline_status():
    """检查数据流水线状态。"""
    status = {
        "classification_data": (config.data_dir / "rssi_processed_classification.pkl").exists(),
        "identification_data": (config.data_dir / "rssi_processed_identification.pkl").exists(),
        "classification_model": (config.model_dir / "best_classification_model.joblib").exists(),
        "identification_model": (config.model_dir / "identification_model.joblib").exists(),
        "cnn_classification_model": (config.model_dir / "cnn_classification.pt").exists(),
        "cnn_identification_model": (config.model_dir / "cnn_identification.pt").exists(),
    }
    return status


def display_pipeline_status():
    """显示流水线状态。"""
    status = check_data_pipeline_status()
    
    st.subheader("流水线状态监控")
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("分类数据", "✓" if status["classification_data"] else "✗")
        st.metric("识别数据", "✓" if status["identification_data"] else "✗")
    
    with cols[1]:
        st.metric("分类模型", "✓" if status["classification_model"] else "✗")
        st.metric("识别模型", "✓" if status["identification_model"] else "✗")
    
    with cols[2]:
        st.metric("CNN分类模型", "✓" if status["cnn_classification_model"] else "✗")
        st.metric("CNN识别模型", "✓" if status["cnn_identification_model"] else "✗")


# 主页面布局
st.title("RSSI 身份识别与分类系统")
st.caption("工作流: 训练 + 推理 + 数据可视化")

page = st.sidebar.radio("页面导航", ["训练页", "推理页", "数据可视化页"])

if page == "训练页":
    st.subheader("模型训练与评估")
    
    # 显示流水线状态
    display_pipeline_status()
    
    # 侧边栏参数配置
    with st.sidebar:
        st.markdown("---")
        st.header("训练参数")
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)
        window_size = st.number_input("窗口大小", min_value=50, max_value=2000, value=200, step=10)
        step_size = st.number_input("滑窗步长", min_value=10, max_value=1000, value=100, step=10)
        use_pca = st.checkbox("启用 PCA", value=True)
        pca_variance = st.slider(
            "PCA 方差保留",
            min_value=0.70,
            max_value=0.99,
            value=0.9019,
            step=0.005,
            disabled=not use_pca,
            help="关闭 PCA 时该参数不生效",
        )
        target_subject = st.text_input("识别目标人员（可选）", value="")
        
        st.markdown("---")
        st.header("训练选项")
        train_mode = st.radio("训练模式", ["增量训练", "完整重训"], index=0)
        train_task = st.radio("训练任务", ["分类", "识别", "CNN分类", "CNN识别", "全部"], index=4)
        
        st.markdown("---")
        st.header("1D CNN 训练参数")
        cnn_epochs = st.number_input("CNN 训练轮数", min_value=1, max_value=200, value=20, step=1)
        cnn_batch_size = st.number_input("CNN 批大小", min_value=8, max_value=512, value=64, step=8)
        cnn_learning_rate = st.number_input("CNN 学习率", min_value=0.00001, max_value=0.1, value=0.001, step=0.0005, format="%.5f")
        cnn_val_ratio = st.slider("CNN 验证集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        cnn_early_stop_patience = st.number_input("CNN 早停耐心轮数", min_value=1, max_value=20, value=5, step=1)

    # 数据预处理区域
    with st.expander("数据预处理", expanded=True):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            mat_files = sorted(config.raw_dir.glob("*.mat"))
            st.write(f"原始文件数: {len(mat_files)}")
            if mat_files:
                st.write("示例文件:", [p.name for p in mat_files[:5]])
        
        with col_b:
            status = check_data_pipeline_status()
            st.write(f"分类数据: {'✓' if status['classification_data'] else '✗'}")
            st.write(f"识别数据: {'✓' if status['identification_data'] else '✗'}")
            st.write(f"分类模型: {'✓' if status['classification_model'] else '✗'}")
            st.write(f"识别模型: {'✓' if status['identification_model'] else '✗'}")
        
        # 数据流水线执行按钮
        run_data_button = st.button("执行数据流水线", type="secondary")
        
        if run_data_button:
            with st.spinner("正在运行数据流水线，请稍候..."):
                # 执行分类数据流水线
                class_result = run_classification_data_pipeline(
                    test_size=float(test_size),
                    seed=int(seed),
                    window_size=int(window_size),
                    step_size=int(step_size),
                    pca_variance=float(pca_variance),
                    use_pca=bool(use_pca),
                )
                
                # 执行识别数据流水线
                id_result = run_identification_data_pipeline(
                    test_size=float(test_size),
                    seed=int(seed),
                    window_size=int(window_size),
                    step_size=int(step_size),
                    pca_variance=float(pca_variance),
                    use_pca=bool(use_pca),
                )

            st.success("分类与识别数据流水线执行完成")
            
            st.subheader("分类数据流程元信息")
            st.json({
                "split": class_result["split_meta"],
                "window": class_result["window_meta"],
                "process": class_result["process_meta"],
            })
            
            st.subheader("识别数据流程元信息")
            st.json({
                "split": id_result["split_meta"],
                "window": id_result["window_meta"],
                "process": id_result["process_meta"],
            })

    # 模型训练区域
    with st.expander("模型训练", expanded=True):
        train_col1, train_col2, train_col3, train_col4 = st.columns(4)
        
        with train_col1:
            if st.button("训练分类模型", type="primary", key="train_class"):
                if train_mode == "完整重训":
                    # 重新执行数据流水线
                    with st.spinner("重新执行数据流水线..."):
                        run_classification_data_pipeline(
                            test_size=float(test_size),
                            seed=int(seed),
                            window_size=int(window_size),
                            step_size=int(step_size),
                            pca_variance=float(pca_variance),
                            use_pca=bool(use_pca),
                        )
                
                with st.spinner("正在训练身份分类模型..."):
                    class_result = ModelTrainingPipeline().run_classification(
                        data_file=config.data_dir / "rssi_processed_classification.pkl"
                    )
                
                st.success("身份分类训练完成")
                class_metrics_payload = {
                    name: metrics
                    for name, metrics in class_result.items()
                    if isinstance(metrics, dict)
                }
                class_metrics = pd.DataFrame(cast(dict[str, Any], class_metrics_payload)).T
                st.subheader("分类任务指标对比")
                st.dataframe(class_metrics, width="stretch")
        
        with train_col2:
            if st.button("训练识别模型", type="primary", key="train_id"):
                if train_mode == "完整重训":
                    # 重新执行数据流水线
                    with st.spinner("重新执行数据流水线..."):
                        run_identification_data_pipeline(
                            test_size=float(test_size),
                            seed=int(seed),
                            window_size=int(window_size),
                            step_size=int(step_size),
                            pca_variance=float(pca_variance),
                            use_pca=bool(use_pca),
                        )
                
                with st.spinner("正在训练身份识别模型..."):
                    id_result = ModelTrainingPipeline().run_identification(
                        data_file=config.data_dir / "rssi_processed_identification.pkl",
                    )
                
                st.success("身份识别训练完成")
                id_metrics_payload = id_result.get("metrics", {}) if isinstance(id_result, dict) else {}
                st.subheader("识别任务指标")
                st.write(f"拒识阈值: {id_result.get('threshold', 0):.4f}")
                st.dataframe(pd.DataFrame(cast(dict[str, Any], id_metrics_payload)).T, width="stretch")
        
        with train_col3:
            if st.button("训练CNN分类模型", type="primary", key="train_cnn_class"):
                if train_mode == "完整重训":
                    # 重新执行数据流水线
                    with st.spinner("重新执行数据流水线..."):
                        run_classification_data_pipeline(
                            test_size=float(test_size),
                            seed=int(seed),
                            window_size=int(window_size),
                            step_size=int(step_size),
                            pca_variance=float(pca_variance),
                            use_pca=bool(use_pca),
                        )
                
                with st.spinner("正在训练 1D CNN 分类模型..."):
                    cnn_class_trainer = _build_cnn_trainer(
                        epochs=int(cnn_epochs),
                        batch_size=int(cnn_batch_size),
                        learning_rate=float(cnn_learning_rate),
                        val_ratio=float(cnn_val_ratio),
                        early_stop_patience=int(cnn_early_stop_patience),
                        seed=int(seed),
                    )
                    cnn_class_result = cnn_class_trainer.train_classification(
                        data_file=config.data_dir / "rssi_windowed_classification.pkl",
                    )
                
                st.success("1D CNN 分类训练完成")
                st.json(cnn_class_result)

        with train_col4:
            if st.button("训练CNN识别模型", type="primary", key="train_cnn_id"):
                if train_mode == "完整重训":
                    # 重新执行识别数据流水线
                    with st.spinner("重新执行识别数据流水线..."):
                        run_identification_data_pipeline(
                            test_size=float(test_size),
                            seed=int(seed),
                            window_size=int(window_size),
                            step_size=int(step_size),
                            pca_variance=float(pca_variance),
                            use_pca=bool(use_pca),
                        )

                with st.spinner("正在训练 1D CNN 识别模型..."):
                    cnn_id_trainer = _build_cnn_trainer(
                        epochs=int(cnn_epochs),
                        batch_size=int(cnn_batch_size),
                        learning_rate=float(cnn_learning_rate),
                        val_ratio=float(cnn_val_ratio),
                        early_stop_patience=int(cnn_early_stop_patience),
                        seed=int(seed),
                    )
                    cnn_id_result = cnn_id_trainer.train_identification(
                        data_file=config.data_dir / "rssi_windowed_identification.pkl",
                    )

                st.success("1D CNN 识别训练完成")
                st.json(cnn_id_result)

    # 批量训练选项
    with st.expander("批量训练", expanded=False):
        if st.button("执行批量训练", type="secondary"):
            tasks_to_run = []
            if train_task in ["分类", "全部"]:
                tasks_to_run.append(("分类", lambda: ModelTrainingPipeline().run_classification(data_file=config.data_dir / "rssi_processed_classification.pkl")))
            if train_task in ["识别", "全部"]:
                tasks_to_run.append(("识别", lambda: ModelTrainingPipeline().run_identification(
                    data_file=config.data_dir / "rssi_processed_identification.pkl")))
            if train_task in ["CNN分类", "全部"]:
                tasks_to_run.append(("CNN分类", lambda: _build_cnn_trainer(
                    epochs=int(cnn_epochs),
                    batch_size=int(cnn_batch_size),
                    learning_rate=float(cnn_learning_rate),
                    val_ratio=float(cnn_val_ratio),
                    early_stop_patience=int(cnn_early_stop_patience),
                    seed=int(seed),
                ).train_classification(data_file=config.data_dir / "rssi_windowed_classification.pkl")))
            if train_task in ["CNN识别", "全部"]:
                tasks_to_run.append(("CNN识别", lambda: _build_cnn_trainer(
                    epochs=int(cnn_epochs),
                    batch_size=int(cnn_batch_size),
                    learning_rate=float(cnn_learning_rate),
                    val_ratio=float(cnn_val_ratio),
                    early_stop_patience=int(cnn_early_stop_patience),
                    seed=int(seed),
                ).train_identification(data_file=config.data_dir / "rssi_windowed_identification.pkl")))

            for task_name, task_func in tasks_to_run:
                with st.spinner(f"正在训练{task_name}模型..."):
                    try:
                        result = task_func()
                        st.success(f"{task_name}模型训练完成")
                    except Exception as e:
                        st.error(f"{task_name}模型训练失败: {str(e)}")

    # 模型评估区域
    st.divider()
    st.subheader("模型评估")
    render_confusion_and_importance()

elif page == "推理页":
    st.subheader("单文件推理")
    payload = load_processed_payload("rssi_processed_classification.pkl")
    if payload is None:
        st.warning("请先在训练页执行完整流水线")
    else:
        # 加载模型和预处理器
        pca = payload["pca"]
        scaler = payload["scaler"]
        meta = payload.get("meta", {})
        use_pca_in_model = bool(meta.get("use_pca", True))
        low_freq_bins = int(meta.get("low_freq_bins", 16))
        encoder = LabelEncoder()
        encoder.fit(np.asarray(payload["y_train"], dtype=object))

        # 左侧参数选择
        col_l, col_r = st.columns([1, 1])
        with col_l:
            source_mode = st.radio("输入来源", ["选择 raw 文件", "上传 .mat 文件"])
            window_size = st.number_input("推理窗口大小", min_value=50, max_value=2000, value=200, step=10)
            step_size = st.number_input("推理滑窗步长", min_value=10, max_value=1000, value=100, step=10)
            task_type = st.radio("任务类型", ["身份分类", "身份识别"])
            model_family = st.radio("模型类型", ["传统模型", "1D CNN"])

        # 右侧数据输入
        matrix: np.ndarray | None = None
        with col_r:
            if source_mode == "选择 raw 文件":
                raw_files = sorted(config.raw_dir.glob("*.mat"))
                file_name = st.selectbox("选择文件", [p.name for p in raw_files]) if raw_files else None
                if file_name is not None:
                    mat_obj = loadmat(config.raw_dir / file_name)
                    matrix = get_rssi_from_mat_object(mat_obj)
            else:
                uploaded = st.file_uploader("上传 MAT 文件", type=["mat"])
                if uploaded is not None:
                    mat_obj = loadmat(uploaded)
                    matrix = get_rssi_from_mat_object(mat_obj)

            if matrix is not None:
                st.write(f"输入形状: {matrix.shape}")

        # 执行推理
        if matrix is not None and st.button("开始推理", type="primary"):
            try:
                if model_family == "传统模型":
                    if task_type == "身份分类":
                        # 分类任务推理
                        x_infer = build_inference_feature(
                            matrix=matrix,
                            window_size=int(window_size),
                            step_size=int(step_size),
                            low_freq_bins=low_freq_bins,
                            pca=pca,
                            scaler=scaler,
                            use_pca=use_pca_in_model,
                        )
                        
                        # 验证模型兼容性
                        model_path = config.model_dir / "best_classification_model.joblib"
                        is_compatible, error_msg = validate_data_model_compatibility("classification", model_path, payload)
                        if not is_compatible:
                            st.error(error_msg)
                            st.stop()
                        
                        with st.spinner("正在推理..."):
                            model = joblib.load(model_path)
                            pred_encoded = np.asarray(model.predict(x_infer), dtype=int)
                            pred_label = encoder.inverse_transform(pred_encoded)
                            vote = pd.Series(pred_label).value_counts(normalize=True)
                            top_label = vote.index[0]
                            top_score = float(vote.iloc[0])
                            st.success(f"预测身份: {top_label} (置信度: {top_score:.2%})")
                            st.dataframe(vote.rename("占比").to_frame())
                    else:
                        # 识别任务推理
                        id_payload = load_processed_payload("rssi_processed_identification.pkl")
                        if id_payload is None:
                            st.error("未找到识别任务处理数据，请先在训练页执行身份识别数据流水线")
                            raise ValueError("识别处理数据缺失")
                        
                        id_pca = id_payload["pca"]
                        id_scaler = id_payload["scaler"]
                        id_use_pca = bool(id_payload.get("meta", {}).get("use_pca", True))
                        id_low_freq_bins = int(id_payload.get("meta", {}).get("low_freq_bins", 16))
                        x_infer = build_inference_feature(
                            matrix=matrix,
                            window_size=int(window_size),
                            step_size=int(step_size),
                            low_freq_bins=id_low_freq_bins,
                            pca=id_pca,
                            scaler=id_scaler,
                            use_pca=id_use_pca,
                        )
                        
                        # 验证模型兼容性
                        model_path = config.model_dir / "identification_model.joblib"
                        is_compatible, error_msg = validate_data_model_compatibility("identification", model_path, id_payload)
                        if not is_compatible:
                            st.error(error_msg)
                            st.stop()
                        
                        with st.spinner("正在推理..."):
                            loaded = joblib.load(model_path)
                            if isinstance(loaded, dict) and "centroids" in loaded:
                                known_labels = np.asarray(loaded.get("known_labels", []), dtype=object)
                                centroids = np.asarray(loaded.get("centroids", []), dtype=np.float32)
                                threshold = float(loaded.get("threshold", 0.0))
                                unknown_label = str(loaded.get("unknown_label", "Unknown"))

                                if centroids.ndim != 2 or known_labels.shape[0] != centroids.shape[0]:
                                    raise ValueError("识别模型元数据损坏：中心向量或标签数量不一致")

                                distances = np.linalg.norm(x_infer[:, None, :] - centroids[None, :, :], axis=2)
                                min_dist = np.min(distances, axis=1)
                                pred_idx = np.argmin(distances, axis=1)
                                nearest_pred = known_labels[pred_idx]
                                open_pred = np.asarray(
                                    [
                                        str(nearest_pred[i]) if float(min_dist[i]) <= threshold else unknown_label
                                        for i in range(nearest_pred.shape[0])
                                    ],
                                    dtype=object,
                                )

                                vote = pd.Series(open_pred).value_counts(normalize=True)
                                top_label = str(vote.index[0])
                                top_score = float(vote.iloc[0])
                                unknown_ratio = float(vote.get(unknown_label, 0.0))
                                st.success(f"识别结果: {top_label} (窗口占比: {top_score:.2%})")
                                st.write(f"Unknown 窗口占比: {unknown_ratio:.2%}")
                                st.write(f"距离拒识阈值: {threshold:.4f}")
                                st.write(f"平均最小距离: {float(np.mean(min_dist)):.4f}")
                                st.dataframe(vote.rename("占比").to_frame())
                            elif isinstance(loaded, dict) and "model" in loaded:
                                model = loaded["model"]
                                threshold = float(loaded.get("threshold", 0.5))
                                known_labels = np.asarray(loaded.get("known_labels", []), dtype=object)
                                unknown_label = str(loaded.get("unknown_label", "Unknown"))

                                proba = np.asarray(model.predict_proba(x_infer), dtype=np.float32)
                                max_score = np.max(proba, axis=1)
                                pred_idx = np.argmax(proba, axis=1)
                                if known_labels.shape[0] == 0:
                                    known_pred = np.asarray(["Unknown"] * x_infer.shape[0], dtype=object)
                                else:
                                    known_pred = known_labels[pred_idx]
                                open_pred = np.asarray(
                                    [
                                        str(known_pred[i]) if float(max_score[i]) >= threshold else unknown_label
                                        for i in range(known_pred.shape[0])
                                    ],
                                    dtype=object,
                                )
                                vote = pd.Series(open_pred).value_counts(normalize=True)
                                top_label = str(vote.index[0])
                                top_score = float(vote.iloc[0])
                                unknown_ratio = float(vote.get(unknown_label, 0.0))
                                st.success(f"识别结果: {top_label} (窗口占比: {top_score:.2%})")
                                st.write(f"Unknown 窗口占比: {unknown_ratio:.2%}")
                                st.write(f"开集拒识阈值: {threshold:.4f}")
                                st.dataframe(vote.rename("占比").to_frame())
                            else:
                                model_any: Any = loaded
                                ocsvm_pred = np.asarray(model_any.predict(x_infer), dtype=int)
                                pred = np.where(ocsvm_pred == -1, 1, 0)
                                positive_ratio = float(np.mean(pred))
                                st.warning("当前识别模型为旧版二分类逻辑，建议重新训练以启用 ID/Unknown 开集识别")
                                st.success(f"未见人员窗口占比: {positive_ratio:.2%}")
                else:
                    # CNN模型推理
                    cnn_window_size = int(window_size)
                    cnn_windows = build_cnn_inference_windows(matrix=matrix, window_size=cnn_window_size, step_size=int(step_size))
                    
                    if task_type == "身份分类":
                        # CNN分类推理
                        cnn_path = config.model_dir / "cnn_classification.pt"
                        if not cnn_path.exists():
                            st.error("未找到 1D CNN 分类模型，请先训练")
                        else:
                            with st.spinner("正在推理..."):
                                cnn_predictor = CNNInference(cnn_path, device="cpu")
                                cnn_pred_label = np.asarray(cnn_predictor.predict(cnn_windows), dtype=object)
                                vote = pd.Series(cnn_pred_label).value_counts(normalize=True)
                                top_label = vote.index[0]
                                top_score = float(vote.iloc[0])
                                st.success(f"CNN 预测身份: {top_label} (置信度: {top_score:.2%})")
                                st.dataframe(vote.rename("占比").to_frame())
                    else:
                        # CNN识别推理
                        cnn_path = config.model_dir / "cnn_identification.pt"
                        if not cnn_path.exists():
                            st.error("未找到 1D CNN 识别模型，请先训练")
                        else:
                            with st.spinner("正在推理..."):
                                cnn_predictor = CNNInference(cnn_path, device="cpu")
                                open_pred = np.asarray(cnn_predictor.predict_open_set(cnn_windows), dtype=object)
                                vote = pd.Series(open_pred).value_counts(normalize=True)
                                top_label = str(vote.index[0])
                                top_score = float(vote.iloc[0])
                                unknown_ratio = float(vote.get("Unknown", 0.0))
                                st.success(f"CNN 识别结果: {top_label} (窗口占比: {top_score:.2%})")
                                st.write(f"CNN Unknown 窗口占比: {unknown_ratio:.2%}")
                                st.dataframe(vote.rename("占比").to_frame())

            except Exception as exc:
                st.error(f"推理失败: {exc}")

else:
    st.subheader("数据可视化")
    raw_files = sorted(config.raw_dir.glob("*.mat"))
    if not raw_files:
        st.warning("raw 目录为空")
    else:
        # 选择并加载文件
        selected = st.selectbox("选择原始文件", [p.name for p in raw_files])
        mat_obj = loadmat(config.raw_dir / selected)
        matrix = get_rssi_from_mat_object(mat_obj)

        st.write(f"数据形状: {matrix.shape}")
        st.line_chart(pd.DataFrame(matrix[:1000, :5], columns=[f"特征维{i+1}" for i in range(5)]))

        # 绘制热力图
        fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
        sns.heatmap(matrix[:400, :].T, cmap="mako", ax=ax_heat)
        ax_heat.set_title("RSSI 热力图（前 400 时间点）")
        ax_heat.set_xlabel("时间")
        ax_heat.set_ylabel("特征维")
        st.pyplot(fig_heat)

        # 绘制直方图
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        ax_hist.hist(matrix[:, 0], bins=40, color="#2a9d8f", alpha=0.85)
        ax_hist.set_title("特征维1 RSSI 分布")
        ax_hist.set_xlabel("RSSI")
        ax_hist.set_ylabel("频数")
        st.pyplot(fig_hist)

    # 显示处理后数据的可视化
    payload = load_processed_payload()
    if payload is not None:
        st.divider()
        use_pca_in_model = bool(payload.get("meta", {}).get("use_pca", True))
        st.subheader("PCA 特征分布" if use_pca_in_model else "归一化特征分布（PCA关闭）")
        x_train = np.asarray(payload["x_train"], dtype=np.float32)
        y_train = np.asarray(payload["y_train"], dtype=object)
        x_test = np.asarray(payload["x_test"], dtype=np.float32)
        y_test = np.asarray(payload["y_test"], dtype=object)

        # 准备绘图数据
        train_df = pd.DataFrame(
            {
                "pc1": x_train[:, 0],
                "pc2": x_train[:, 1] if x_train.shape[1] > 1 else np.zeros_like(x_train[:, 0]),
                "label": y_train,
                "split": "train",
            }
        )
        test_df = pd.DataFrame(
            {
                "pc1": x_test[:, 0],
                "pc2": x_test[:, 1] if x_test.shape[1] > 1 else np.zeros_like(x_test[:, 0]),
                "label": y_test,
                "split": "test",
            }
        )
        draw_df = pd.concat([train_df.sample(min(1500, len(train_df)), random_state=42), test_df], axis=0)

        # 绘制PCA散点图
        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=draw_df, x="pc1", y="pc2", hue="label", style="split", alpha=0.7, s=35, ax=ax_pca)
        ax_pca.set_title("PCA 空间分布")
        st.pyplot(fig_pca)
