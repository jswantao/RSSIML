# -*- coding: utf-8 -*-
"""实验模块 — E1~E5 论文实验的完整实现。

用法:
    from experiments import ExperimentRunner, render_e1_results, ...

    runner = ExperimentRunner("csi", check_cancelled=_check_cancelled)
    runner.run_e1()
    print(runner.results)
"""
from experiments.base import BaseExperimentRunner
from experiments.e1_csi import run_e1, render_e1_results
from experiments.e2_rssi import run_e2, render_e2_results
from experiments.e3_enrollment import run_e3, render_e3_results
from experiments.e4_compare import run_e4, render_e4_results
from experiments.e5_rssi_slice import run_e5, render_e5_results


class ExperimentRunner(
    BaseExperimentRunner,
):
    """完整实验执行器 — 继承 Base + 各实验 Mixin。"""

    run_e1 = run_e1
    run_e2 = run_e2
    run_e3 = run_e3
    run_e4 = run_e4
    run_e5 = run_e5


__all__ = [
    "ExperimentRunner",
    "render_e1_results",
    "render_e2_results",
    "render_e3_results",
    "render_e4_results",
    "render_e5_results",
]
