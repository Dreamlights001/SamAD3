import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_sam3_detector
from dataset import get_anomaly_dataset
from utils.prompt_engineering import get_prompt_generator

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3异常检测模型微调')
    parser.add_argument('--model_path', type=str, default='~/autodl-tmp/pre-training/sam3.pth', help='SAM3模型权重路径')
    parser.add_argument('--dataset_root', type=str, default='~/autodl-tmp/datasets', help='数据集根目录')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['mvtec', 'visa'], help='数据集名称')
    parser.add_argument('--category', type=str, required=True, help='数据集类别')
    parser.add_argument('--output_dir', type=str, default='../assets/experiments', help='输出结果目录')
    parser.add_argument('--checkpoint_dir', type=str, default='../assets/checkpoints', help='检查点保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'], help='学习率调度器')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--finetune_layers', type=str, default='prompt_encoder', choices=['prompt_encoder', 'mask_decoder', 'both'], help='微调的层')
    parser.add_argument('--augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--custom_prompts', type=str, default=None, help='自定义提示词文件路径')
    parser.add_argument('--save_interval', type=int, default=10, help='检查点保存间隔')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    return parser.parse_args()

def get_transforms(augmentation=False):
    """
    获取图像变换
    """
    transform_list = [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
    
    if augmentation:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(1, transforms.RandomRotation(10))
        transform_list.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    
    return transforms.Compose(transform_list)

def get_target_transforms():
    """
    获取目标变换
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

def get_loss_function():
    """
    获取损失函数
    """
    # 分割损失：二元交叉熵 + Dice损失
    bce_loss = nn.BCELoss()
    
    def dice_loss(pred, target, smooth=1e-6):
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def combined_loss(pred, target):
        bce = bce_loss(pred, target)
        dice = dice_loss(pred, target)
        return bce + dice
    
    return combined_loss

def configure_optimizer(model, args):
    """
    配置优化器
    """
    # 根据选择的微调层确定需要优化的参数
    if args.finetune_layers == 'prompt_encoder':
        params_to_optimize = list(model.prompt_encoder.parameters())
    elif args.finetune_layers == 'mask_decoder':
        params_to_optimize = list(model.mask_decoder.parameters())
    else:  # both
        params_to_optimize = list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 配置学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - args.warmup_epochs, 
            eta_min=1e-7
        )
    
    return optimizer, scheduler

def train_one_epoch(model, dataloader, loss_fn, optimizer, prompt_gen, args, epoch, writer):
    """
    训练一个轮次
    """
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}'):
        images = batch['image'].to(args.device)
        gt_masks = batch.get('mask')
        image_paths = batch['image_path']
        
        optimizer.zero_grad()
        
        batch_loss = 0.0
        
        for i in range(len(images)):
            # 将tensor转换回PIL图像
            img_tensor = images[i].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            
            # 获取该类别的提示词
            prompts = prompt_gen.get_dataset_prompts(args.dataset_name, args.category)
            
            # 执行异常检测
            results = model.detect_anomalies(
                image=img_pil,
                text_prompts=prompts,
                threshold=0.5,
                return_logits=True  # 返回logits用于训练
            )
            
            # 计算损失
            if gt_masks is not None and len(results['masks']) > 0:
                # 选择分数最高的掩码
                best_mask_idx = results['scores'].argmax()
                pred_mask = results['logits'][best_mask_idx].squeeze(0)
                
                # 确保形状匹配
                target_mask = gt_masks[i].to(args.device).squeeze(0)
                
                # 应用sigmoid激活函数
                pred_mask = torch.sigmoid(pred_mask)
                
                # 计算损失
                loss = loss_fn(pred_mask, target_mask)
                batch_loss += loss
            
        # 平均批次损失
        if len(images) > 0:
            batch_loss /= len(images)
            batch_loss.backward()
            optimizer.step()
            
            # 更新累计损失
            epoch_loss += batch_loss.item() * len(images)
            total_samples += len(images)
    
    # 计算平均轮次损失
    avg_epoch_loss = epoch_loss / total_samples
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.8f}')
    
    return avg_epoch_loss

def warmup_lr_scheduler(optimizer, warmup_epochs, epoch, total_epochs, start_lr):
    """
    学习率预热调度器
    """
    if epoch < warmup_epochs:
        lr = start_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return True
    return False

def save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    """
    保存检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'args': vars(args)
    }
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'检查点已保存至: {checkpoint_path}')

def run_training(args):
    """
    运行训练
    """
    # 设置随机种子确保可复现性
    torch.manual_seed(122)
    np.random.seed(122)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 加载数据集
    print(f"加载数据集: {args.dataset_name}/{args.category}")
    dataset = get_anomaly_dataset(
        root_dir=os.path.join(args.dataset_root, args.dataset_name),
        dataset_name=args.dataset_name,
        category=args.category,
        split='train',
        transform=get_transforms(args.augmentation),
        target_transform=get_target_transforms(),
        load_masks=True
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 加载提示词生成器
    prompt_gen = get_prompt_generator(args.custom_prompts)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = get_sam3_detector(args.model_path)
    model.to(args.device)
    
    # 配置优化器和学习率调度器
    optimizer, scheduler = configure_optimizer(model, args)
    
    # 加载损失函数
    loss_fn = get_loss_function()
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"恢复到轮次: {start_epoch}")
    
    # 开始训练
    print("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # 学习率预热
        if warmup_lr_scheduler(optimizer, args.warmup_epochs, epoch, args.epochs, args.learning_rate):
            pass  # 预热期间不使用调度器
        elif epoch >= args.warmup_epochs and scheduler:
            scheduler.step()
        
        # 训练一个轮次
        epoch_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, prompt_gen, args, epoch, writer)
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'最佳模型已保存至: {best_model_path}')
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_loss, args)
    
    # 训练结束，保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存至: {final_model_path}')
    
    # 关闭TensorBoard记录器
    writer.close()

def main():
    # 解析参数
    args = parse_args()
    
    # 运行训练
    run_training(args)

if __name__ == '__main__':
    import sys
    main()
