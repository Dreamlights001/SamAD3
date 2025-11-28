#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3异常检测工具包

该包包含了SAM3异常检测项目所需的各种工具函数，包括：
- 评估指标计算（AUROC、AP、F1_Score、AUPRO等）
- 提示工程工具
- 数据处理工具
- 可视化工具
- 日志工具
"""

from .evaluation import (
    compute_image_level_auroc,
    compute_pixel_level_auroc,
    compute_pixel_level_aupro,
    compute_pixel_level_ap,
    compute_f1_score,
    compute_evaluation_metrics,
    compute_segmentation_metrics,
    normalize_anomaly_scores,
    apply_gaussian_smoothing,
    calculate_optimal_threshold
)

from .prompt_engineering import (
    AnomalyPromptGenerator,
    get_prompt_generator
)

# from .visualization import (
#     visualize_mask_overlay,
#     create_comparison_plot,
#     plot_roc_curve,
#     plot_pr_curve,
#     create_tsne_visualization,
#     create_result_grid
# )

# from .data_utils import (
#     convert_to_rgb,
#     resize_and_pad,
#     normalize_tensor,
#     denormalize_tensor,
#     apply_color_map,
#     compute_image_stats
# )

# from .logger import (
#     setup_logger,
#     get_logger,
#     print_metrics_summary
# )

__all__ = [
    # 评估指标
    'compute_image_level_auroc',
    'compute_pixel_level_auroc',
    'compute_pixel_level_aupro',
    'compute_pixel_level_ap',
    'compute_f1_score',
    'compute_evaluation_metrics',
    'compute_segmentation_metrics',
    'normalize_anomaly_scores',
    'apply_gaussian_smoothing',
    'calculate_optimal_threshold',
    
    # 提示工程
    'AnomalyPromptGenerator',
    'get_prompt_generator'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "SAM3异常检测团队"