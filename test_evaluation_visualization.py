#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试评估指标和可视化功能

本脚本用于测试新添加的评估指标和可视化功能是否正常工作。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from utils.metrics import (
    calculate_image_level_auroc,
    calculate_pixel_level_auroc,
    calculate_ap,
    calculate_aupro,
    evaluate_batch_predictions
)
from utils.image_processing import (
    visualize_prediction,
    create_visualization_grid,
    save_visualization
)

def test_metrics():
    """测试评估指标计算功能"""
    print("===== 测试评估指标计算 =====")
    
    # 创建测试数据
    # 3张测试图像的预测掩码和真实掩码
    pred_masks = [
        np.zeros((100, 100)),  # 无异常
        np.ones((100, 100)) * 0.5,  # 全图中等置信度
        np.zeros((100, 100))
    ]
    pred_masks[2][20:40, 20:40] = 1.0  # 局部高置信度
    
    true_masks = [
        np.zeros((100, 100)),  # 无异常
        np.ones((100, 100)),  # 全图异常
        np.zeros((100, 100))
    ]
    true_masks[2][30:50, 30:50] = 1.0  # 局部异常（与预测有部分重叠）
    
    # 计算各项指标
    print("计算图像级AUROC:")
    image_auroc = calculate_image_level_auroc(pred_masks, true_masks)
    print(f"图像级AUROC: {image_auroc:.4f}")
    
    print("\n计算像素级AUROC:")
    pixel_auroc = calculate_pixel_level_auroc(pred_masks, true_masks)
    print(f"像素级AUROC: {pixel_auroc:.4f}")
    
    print("\n计算AP:")
    ap = calculate_ap(pred_masks, true_masks)
    print(f"AP: {ap:.4f}")
    
    print("\n计算AUPRO:")
    aupro = calculate_aupro(pred_masks, true_masks)
    print(f"AUPRO: {aupro:.4f}")
    
    print("\n计算批量评估指标:")
    all_metrics = evaluate_batch_predictions(pred_masks, true_masks)
    for key, value in all_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n评估指标测试完成！")
    return all_metrics

def test_visualization():
    """测试可视化功能"""
    print("\n===== 测试可视化功能 =====")
    
    # 创建测试图像
    # 创建一个简单的测试图像
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
    # 添加一些简单的图形
    image[50:150, 50:150, :] = [255, 255, 0]  # 黄色方块
    image[75:125, 75:125, :] = [255, 0, 0]   # 红色方块作为"缺陷"
    
    # 创建预测掩码和真实掩码
    pred_mask = np.zeros((200, 200))
    pred_mask[45:155, 45:155] = 0.8  # 预测的缺陷区域略大
    
    true_mask = np.zeros((200, 200))
    true_mask[50:150, 50:150] = 1.0  # 真实的缺陷区域
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试基础可视化
    print("测试基础可视化...")
    basic_viz = visualize_prediction(image, pred_mask)
    basic_viz_path = os.path.join(output_dir, 'basic_visualization.png')
    Image.fromarray(basic_viz).save(basic_viz_path)
    print(f"基础可视化保存至: {basic_viz_path}")
    
    # 测试对比可视化
    print("测试对比可视化...")
    compare_viz = visualize_prediction(image, pred_mask, true_mask)
    compare_viz_path = os.path.join(output_dir, 'compare_visualization.png')
    Image.fromarray(compare_viz).save(compare_viz_path)
    print(f"对比可视化保存至: {compare_viz_path}")
    
    # 测试高级可视化网格
    print("测试高级可视化网格...")
    fig = create_visualization_grid(image, pred_mask, true_mask)
    grid_viz_path = os.path.join(output_dir, 'visualization_grid.png')
    save_visualization(fig, grid_viz_path)
    print(f"可视化网格保存至: {grid_viz_path}")
    
    print("\n可视化功能测试完成！")
    return output_dir

def test_integration():
    """测试评估指标和可视化的集成"""
    print("\n===== 测试集成功能 =====")
    
    # 创建测试数据
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    image[50:150, 50:150, :] = [255, 255, 0]
    
    pred_mask = np.zeros((200, 200))
    pred_mask[45:155, 45:155] = 0.8
    
    true_mask = np.zeros((200, 200))
    true_mask[50:150, 50:150] = 1.0
    
    # 创建批量数据
    pred_masks = [pred_mask]
    true_masks = [true_mask]
    
    # 计算指标
    metrics = evaluate_batch_predictions(pred_masks, true_masks)
    print("计算的指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 创建带指标的可视化
    print("创建带指标的可视化...")
    fig = create_visualization_grid(image, pred_mask, true_mask)
    
    # 添加指标文本
    metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))])
    fig.suptitle(f'评估指标: {metrics_str}', fontsize=12)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_integration_results')
    os.makedirs(output_dir, exist_ok=True)
    
    grid_viz_path = os.path.join(output_dir, 'integration_visualization.png')
    save_visualization(fig, grid_viz_path)
    print(f"集成可视化保存至: {grid_viz_path}")
    
    print("\n集成测试完成！")

def main():
    """主函数"""
    print("开始测试评估指标和可视化功能...")
    
    # 测试评估指标
    metrics = test_metrics()
    
    # 测试可视化功能
    viz_dir = test_visualization()
    
    # 测试集成功能
    test_integration()
    
    print("\n" + "="*50)
    print("所有测试完成！")
    print(f"评估指标测试结果: {metrics}")
    print(f"可视化测试结果保存在: {viz_dir}")
    print("请检查生成的可视化结果是否符合预期。")
    print("="*50)

if __name__ == "__main__":
    import sys
    main()
