import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import get_sam3_detector
from dataset import get_anomaly_dataset
from utils.prompt_engineering import get_prompt_generator
from utils.evaluation import compute_evaluation_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3异常检测测试')
    parser.add_argument('--model_path', type=str, default='~/autodl-tmp/pre-training/sam3.pth', help='SAM3模型权重路径')
    parser.add_argument('--dataset_root', type=str, default='~/autodl-tmp/datasets', help='数据集根目录')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['mvtec', 'visa'], help='数据集名称')
    parser.add_argument('--category', type=str, required=True, help='数据集类别')
    parser.add_argument('--output_dir', type=str, default='../assets/results', help='输出结果目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='异常掩码阈值')
    parser.add_argument('--custom_prompts', type=str, default=None, help='自定义提示词文件路径')
    parser.add_argument('--context_info', type=str, default=None, help='上下文信息，用于混合提示')
    parser.add_argument('--zeroshot_source', type=str, default=None, help='零样本源数据集')
    parser.add_argument('--zeroshot_source_category', type=str, default=None, help='零样本源数据集类别')
    parser.add_argument('--save_visualizations', action='store_true', help='是否保存可视化结果')
    return parser.parse_args()

def get_transforms():
    """
    获取图像变换
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # 可以根据需要添加更多变换
    ])

def get_target_transforms():
    """
    获取目标变换
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

def generate_test_prompts(args) -> list:
    """
    生成测试用的提示词
    """
    prompt_gen = get_prompt_generator(args.custom_prompts)
    
    if args.zeroshot_source:
        # 零样本场景：使用源数据集的提示词
        prompts = prompt_gen.generate_zeroshot_prompts(
            source_dataset=args.zeroshot_source,
            target_dataset=args.dataset_name,
            source_category=args.zeroshot_source_category,
            target_category=args.category,
            context_info=args.context_info
        )
    else:
        # 常规场景：使用目标数据集的提示词
        base_prompts = prompt_gen.get_dataset_prompts(args.dataset_name, args.category)
        
        if args.context_info:
            prompts = prompt_gen.generate_mixed_prompts(
                base_prompts=base_prompts,
                context_info=args.context_info,
                dataset_name=args.dataset_name,
                category=args.category
            )
        else:
            prompts = base_prompts
    
    return prompts

def run_test(args):
    """
    运行测试
    """
    # 设置随机种子确保可复现性
    torch.manual_seed(122)
    np.random.seed(122)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    print(f"加载数据集: {args.dataset_name}/{args.category}")
    dataset = get_anomaly_dataset(
        root_dir=os.path.join(args.dataset_root, args.dataset_name),
        dataset_name=args.dataset_name,
        category=args.category,
        split='test',
        transform=get_transforms(),
        target_transform=get_target_transforms(),
        load_masks=True
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 生成提示词
    prompts = generate_test_prompts(args)
    print(f"使用提示词数量: {len(prompts)}")
    print(f"提示词示例: {prompts[:5]}...")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    detector = get_sam3_detector(args.model_path)
    detector.to(args.device)
    
    # 存储结果
    all_pred_masks = []
    all_gt_masks = []
    all_pred_scores = []
    all_gt_labels = []
    
    # 运行测试
    print("开始测试...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(args.device)
        gt_masks = batch.get('mask')
        gt_labels = batch['is_anomaly']
        image_paths = batch['image_path']
        
        for i in range(len(images)):
            # 将tensor转换回PIL图像
            img_tensor = images[i].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            
            # 执行异常检测
            results = detector.detect_anomalies(
                image=img_pil,
                text_prompts=prompts,
                threshold=args.threshold
            )
            
            # 存储预测结果
            if len(results['masks']) > 0:
                # 选择分数最高的掩码
                best_mask_idx = results['scores'].argmax()
                pred_mask = results['masks'][best_mask_idx].cpu().numpy()
                pred_score = results['scores'][best_mask_idx].item()
            else:
                pred_mask = np.zeros((512, 512))
                pred_score = 0.0
            
            all_pred_masks.append(pred_mask)
            all_pred_scores.append(pred_score)
            
            # 存储真实标签
            if gt_masks is not None:
                all_gt_masks.append(gt_masks[i].cpu().numpy())
            all_gt_labels.append(gt_labels[i].item())
    
    # 计算评估指标
    print("计算评估指标...")
    metrics = compute_evaluation_metrics(
        pred_masks=all_pred_masks,
        gt_masks=all_gt_masks if all_gt_masks else None,
        pred_scores=all_pred_scores,
        gt_labels=all_gt_labels
    )
    
    # 保存结果
    results_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.category}_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n评估结果:")
    print(f"图像级AUROC: {metrics['image_level_auroc']:.4f}")
    print(f"像素级AUROC: {metrics['pixel_level_auroc']:.4f}")
    if 'pixel_level_aupro' in metrics:
        print(f"像素级AUPRO: {metrics['pixel_level_aupro']:.4f}")
    if 'pixel_level_ap' in metrics:
        print(f"像素级AP: {metrics['pixel_level_ap']:.4f}")
    if 'f1_score' in metrics:
        print(f"F1分数: {metrics['f1_score']:.4f}")
    
    print(f"\n结果已保存至: {results_file}")

def main():
    # 解析参数
    args = parse_args()
    
    # 运行测试
    run_test(args)

if __name__ == '__main__':
    main()