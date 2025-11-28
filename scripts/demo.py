import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import get_sam3_detector
from utils.prompt_engineering import get_prompt_generator

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3异常检测演示')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='~/autodl-tmp/pre-training/sam3.pth', help='SAM3模型权重路径')
    parser.add_argument('--output_dir', type=str, default='../assets/results', help='输出结果目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='异常掩码阈值')
    parser.add_argument('--custom_prompts', type=str, default=None, help='自定义提示词文件路径')
    parser.add_argument('--context_info', type=str, default=None, help='上下文信息，用于混合提示')
    parser.add_argument('--dataset_name', type=str, default=None, help='数据集名称，用于生成特定提示词')
    parser.add_argument('--category', type=str, default=None, help='数据集类别，用于生成特定提示词')
    return parser.parse_args()

def load_image(image_path: str) -> Image.Image:
    """
    加载图像
    """
    image = Image.open(image_path).convert('RGB')
    return image

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    预处理图像
    """
    # 这里可以添加更多的预处理步骤
    # 如调整大小、归一化等
    return image

def generate_prompts(args) -> List[str]:
    """
    生成用于异常检测的提示词
    """
    # 获取提示词生成器
    prompt_gen = get_prompt_generator(args.custom_prompts)
    
    if args.dataset_name and args.category:
        # 使用数据集特定的提示词
        base_prompts = prompt_gen.get_dataset_prompts(args.dataset_name, args.category)
    else:
        # 使用基础提示词
        base_prompts = prompt_gen.get_base_prompts()
    
    # 如果提供了上下文信息，生成混合提示词
    if args.context_info:
        mixed_prompts = prompt_gen.generate_mixed_prompts(
            base_prompts=base_prompts,
            context_info=args.context_info,
            dataset_name=args.dataset_name,
            category=args.category
        )
        return mixed_prompts
    
    return base_prompts

def visualize_results(image: Image.Image, 
                     masks: torch.Tensor, 
                     scores: torch.Tensor, 
                     output_path: str):
    """
    可视化检测结果
    """
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # 掩码
    plt.subplot(1, 3, 2)
    if len(masks) > 0:
        # 选择分数最高的掩码
        best_mask_idx = scores.argmax().item()
        best_mask = masks[best_mask_idx].cpu().numpy()
        plt.imshow(best_mask, cmap='jet')
        plt.title(f'Mask (Score: {scores[best_mask_idx].item():.2f})')
    else:
        plt.imshow(np.zeros(image.size[::-1]), cmap='jet')
        plt.title('No Anomaly Detected')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    if len(masks) > 0:
        best_mask_idx = scores.argmax().item()
        best_mask = masks[best_mask_idx].cpu().numpy()
        plt.imshow(best_mask, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子确保可复现性
    torch.manual_seed(122)
    np.random.seed(122)
    
    # 加载图像
    image = load_image(args.image_path)
    
    # 生成提示词
    prompts = generate_prompts(args)
    print(f"使用提示词: {prompts}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    detector = get_sam3_detector(args.model_path)
    detector.to(args.device)
    
    # 执行异常检测
    print("执行异常检测...")
    results = detector.detect_anomalies(
        image=image,
        text_prompts=prompts,
        threshold=args.threshold
    )
    
    # 显示结果
    has_anomaly = results['has_anomaly'].item() if isinstance(results['has_anomaly'], torch.Tensor) else results['has_anomaly']
    print(f"检测结果: {'异常存在' if has_anomaly else '正常'}")
    
    if has_anomaly:
        print(f"检测到 {len(results['masks'])} 个异常区域")
        print(f"异常分数: {results['scores'].cpu().numpy()}")
    
    # 可视化结果
    output_path = os.path.join(args.output_dir, f"result_{os.path.basename(args.image_path).split('.')[0]}.png")
    visualize_results(
        image=image,
        masks=results['masks'],
        scores=results['scores'],
        output_path=output_path
    )
    print(f"结果已保存至: {output_path}")

if __name__ == '__main__':
    import sys
    main()
