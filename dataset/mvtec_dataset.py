import os
import glob
from typing import List, Dict, Optional
from .base_dataset import AnomalyDataset

class MVTecDataset(AnomalyDataset):
    """
    MVTec异常检测数据集加载器
    支持MVTec AD数据集格式
    """
    
    def __init__(self,
                 root_dir: str,
                 category: str,
                 split: str = 'test',
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 load_masks: bool = True):
        """
        初始化MVTec数据集
        
        Args:
            root_dir: 数据集根目录，例如'~/autodl-tmp/datasets/mvtec'
            category: 数据集类别，如'bottle', 'cable'等
            split: 数据集分割（'train'或'test'）
            transform: 图像变换函数
            target_transform: 目标变换函数
            load_masks: 是否加载掩码
        """
        super().__init__(
            root_dir=root_dir,
            dataset_name='mvtec',
            category=category,
            split=split,
            transform=transform,
            target_transform=target_transform,
            load_masks=load_masks
        )
    
    def _load_samples(self):
        """
        加载MVTec数据集的样本信息
        MVTec格式：root/category/{train,test}/{good,anomaly_type}/
        掩码格式：root/category/ground_truth/anomaly_type/
        """
        # 构建类别目录路径
        category_dir = os.path.join(self.root_dir, self.category)
        
        # 确保目录存在
        if not os.path.isdir(category_dir):
            raise FileNotFoundError(f"类别目录不存在: {category_dir}")
        
        # 构建分割目录路径
        split_dir = os.path.join(category_dir, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"分割目录不存在: {split_dir}")
        
        # 获取所有子目录（good和各种异常类型）
        subdirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for subdir in subdirs:
            # 判断是否为正常样本
            is_anomaly = subdir != 'good'
            
            # 获取该子目录下的所有图像
            image_dir = os.path.join(split_dir, subdir)
            image_paths = glob.glob(os.path.join(image_dir, '*.png')) + \
                         glob.glob(os.path.join(image_dir, '*.jpg')) + \
                         glob.glob(os.path.join(image_dir, '*.jpeg'))
            
            # 排序确保一致性
            image_paths.sort()
            
            for image_path in image_paths:
                # 构建样本信息
                sample_info = {
                    'image_path': image_path,
                    'is_anomaly': is_anomaly,
                    'anomaly_type': subdir if is_anomaly else 'good'
                }
                
                # 如果需要掩码且是异常样本，找到对应的掩码路径
                if self.load_masks and is_anomaly:
                    # 构建掩码路径
                    image_name = os.path.basename(image_path)
                    mask_dir = os.path.join(category_dir, 'ground_truth', subdir)
                    
                    # MVTec的掩码文件格式可能是：image_name_gt.png 或 image_name.png
                    mask_path = os.path.join(mask_dir, image_name.replace('.', '_gt.'))
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(mask_dir, image_name)
                    
                    # 如果找到掩码文件，添加到样本信息
                    if os.path.exists(mask_path):
                        sample_info['mask_path'] = mask_path
                    else:
                        # 掩码不存在，但仍然标记为异常
                        sample_info['mask_path'] = None
                
                # 添加到样本列表
                self.samples.append(sample_info)
    
    def get_category_names(self) -> List[str]:
        """
        获取MVTec数据集中的所有类别名称
        """
        # 列出根目录下的所有子目录
        categories = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d)) and 
                     not d.startswith('.')]
        return categories

# 创建一个工厂函数，用于获取MVTec数据集
def get_mvtec_dataset(
    root_dir: str,
    category: str,
    split: str = 'test',
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None,
    load_masks: bool = True
) -> MVTecDataset:
    """
    获取MVTec数据集实例
    
    Args:
        root_dir: 数据集根目录
        category: 数据集类别
        split: 数据集分割
        transform: 图像变换函数
        target_transform: 目标变换函数
        load_masks: 是否加载掩码
        
    Returns:
        MVTec数据集实例
    """
    return MVTecDataset(
        root_dir=root_dir,
        category=category,
        split=split,
        transform=transform,
        target_transform=target_transform,
        load_masks=load_masks
    )
