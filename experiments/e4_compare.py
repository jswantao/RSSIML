from scripts.config import Defaults
# -*- coding: utf-8 -*-
"""E4: 模型性能对比实验 — SVM vs CNN 认证性能对比。"""
import gc
import time

from scripts.models import clear_gpu_memory


def run_e4(self):
    """E4: 模型性能对比 — SVM (在线linear) vs CNN (1D-CNN)。"""
    self.logger.info("=" * 50)
    self.logger.info("E4: 模型性能对比实验")
    results = {}

    self.logger.info("E4: SVM 训练...")
    t0 = time.time()
    r_svm = self._run_svm(
        cross_activity=False,
        seed=Defaults.SEED,
        test_size=Defaults.TEST_SIZE,
        window_size=Defaults.WINDOW_SIZE,
        step_size=Defaults.STEP_SIZE,
        max_files_per_subject=100,
        threshold_method=Defaults.THRESHOLD_METHOD,
        cv_folds=Defaults.CV_FOLDS,
        use_online_svm=True,
        online_kernel="linear",
        csi_denoise="butterworth",
    )
    svm_dur = time.time() - t0
    results["SVM"] = {
        "HTER": r_svm["system_metrics"]["mean_hter"],
        "FAR": r_svm["system_metrics"]["mean_far"],
        "FRR": r_svm["system_metrics"]["mean_frr"],
        "准确率": r_svm["system_metrics"]["global_accuracy"],
        "HTER_std": r_svm["system_metrics"].get("std_hter", 0),
        "FAR_std": r_svm["system_metrics"].get("std_far", 0),
        "FRR_std": r_svm["system_metrics"].get("std_frr", 0),
        "训练耗时(s)": round(svm_dur, 1),
    }
    self.logger.info(f"SVM 完成: HTER={results['SVM']['HTER']:.4f}, {svm_dur:.1f}s")

    self._check_cancelled()
    clear_gpu_memory(); gc.collect()

    # 速度优化: 轻量架构 + 关 checkpoint + 减 epoch + 大批次
    self.logger.info("E4: CNN 训练 (优化: epochs=10, batch=96, light arch, no_ckpt)...")
    t0 = time.time()
    r_cnn = self._run_cnn(
        epochs=10, batch_size=96, max_files_per_subject=100,
        use_checkpoint=False, gradient_accumulation_steps=1,
        use_cache=True, use_model_cache=True,
        conv_channels=(32, 64, 128, 192), hidden_units=256)
    cnn_dur = time.time() - t0
    results["CNN"] = {
        "HTER": r_cnn["system_metrics"]["mean_hter"],
        "FAR": r_cnn["system_metrics"]["mean_far"],
        "FRR": r_cnn["system_metrics"]["mean_frr"],
        "准确率": r_cnn["system_metrics"]["global_accuracy"],
        "HTER_std": r_cnn["system_metrics"].get("std_hter", 0),
        "FAR_std": r_cnn["system_metrics"].get("std_far", 0),
        "FRR_std": r_cnn["system_metrics"].get("std_frr", 0),
        "训练耗时(s)": round(cnn_dur, 1),
    }
    self.logger.info(f"CNN 完成: HTER={results['CNN']['HTER']:.4f}, {cnn_dur:.1f}s")

    self.results["E4"] = results
    return results


def render_e4_results(results):
    """E4 结果: 模型对比 — 结果有效性校验, 实际渲染由 app_auth._render_compare_results 完成。"""
    if not isinstance(results, dict) or not results:
        return False
    return True
