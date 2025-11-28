"""
异常检测数据集模块

该模块提供了用于加载和处理MVTec和VisA等异常检测数据集的类和工具。
"""

from .base_dataset import AnomalyDataset
from .mvtec_dataset import MVTecDataset, get_mvtec_dataset
from .visa_dataset import VisADataset, get_visa_dataset

# 创建一个统一的数据集工厂函数
def get_anomaly_dataset(
    root_dir: str,
    dataset_name: str,
    category: str,
    split: str = 'test',
    transform = None,
    target_transform = None,
    load_masks: bool = True
):
    """
    获取异常检测数据集实例
    
    Args:
        root_dir: 数据集根目录
        dataset_name: 数据集名称（'mvtec'或'visa'）
        category: 数据集类别
        split: 数据集分割
        transform: 图像变换函数
        target_transform: 目标变换函数
        load_masks: 是否加载掩码
        
    Returns:
        数据集实例
    """
    if dataset_name.lower() == 'mvtec':
        return get_mvtec_dataset(
            root_dir=root_dir,
            category=category,
            split=split,
            transform=transform,
            target_transform=target_transform,
            load_masks=load_masks
        )
    elif dataset_name.lower() == 'visa':
        return get_visa_dataset(
            root_dir=root_dir,
            category=category,
            split=split,
            transform=transform,
            target_transform=target_transform,
            load_masks=load_masks
        )
    else:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")

__all__ = [
    "AnomalyDataset",
    "MVTecDataset",
    "VisADataset",
    "get_mvtec_dataset",
    "get_visa_dataset",
    "get_anomaly_dataset"
]
