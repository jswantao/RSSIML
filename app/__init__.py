# -*- coding: utf-8 -*-
"""WiFiML 身份认证 Streamlit 前端 — 模块化架构。

- state:  Session 状态管理 + 全局配置
- ui:     UI 交互组件 (操作按钮、进度条、字体)
- auth:   认证工具 (模型加载/保存、CSI 拼接、用户映射)
- executor: 训练执行器基类 + SSE/SVM/CNN/实验包装器
- training: 注册阶段渲染 (基础训练/参数研究/模型对比)
- inference: 认证阶段渲染 (单次/持续认证、分数图表)
- experiments: 实验面板渲染
"""
