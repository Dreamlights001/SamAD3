import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class AnomalyDataset(Dataset):
    """
    异常检测数据集的基础类
    为MVTec和VisA等数据集提供统一的接口
    """
    
    def __init__(self,
                 root_dir: str,
                 dataset_name: str,
                 category: str,
                 split: str = 'test',  # 'train' or 'test'
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 load_masks: bool = True):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            dataset_name: 数据集名称（如'mvtec', 'visa'）
            category: 数据集类别
            split: 数据集分割（'train'或'test'）
            transform: 图像变换函数
            target_transform: 目标变换函数
            load_masks: 是否加载掩码
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.category = category
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.load_masks = load_masks
        
        # 存储数据样本的列表
        self.samples = []
        
        # 初始化时加载样本信息
        self._load_samples()
    
    def _load_samples(self):
        """
        加载数据样本信息
        由子类实现
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, str, bool]]:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像、掩码、标签等信息的字典
        """
        sample_info = self.samples[idx]
        
        # 加载图像
        image_path = sample_info['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # 加载掩码（如果需要）
        mask = None
        if self.load_masks and 'mask_path' in sample_info and sample_info['mask_path']:
            mask_path = sample_info['mask_path']
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if mask is not None and self.target_transform:
            mask = self.target_transform(mask)
        
        # 构建返回字典
        result = {
            'image': image,
            'is_anomaly': sample_info['is_anomaly'],
            'image_path': image_path,
            'category': self.category,
            'dataset': self.dataset_name
        }
        
        if mask is not None:
            result['mask'] = mask
            
        if 'anomaly_type' in sample_info:
            result['anomaly_type'] = sample_info['anomaly_type']
        
        return result
    
    def get_normal_samples(self) -> List[Dict]:
        """
        获取所有正常样本
        """
        return [sample for sample in self.samples if not sample['is_anomaly']]
    
    def get_anomaly_samples(self) -> List[Dict]:
        """
        获取所有异常样本
        """
        return [sample for sample in self.samples if sample['is_anomaly']]
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取数据集统计信息
        """
        normal_count = len(self.get_normal_samples())
        anomaly_count = len(self.get_anomaly_samples())
        
        return {
            'total': len(self),
            'normal': normal_count,
            'anomaly': anomaly_count,
            'categories': 1  # 这里是单个类别
        }
    
    def get_anomaly_types(self) -> List[str]:
        """
        获取所有异常类型
        """
        anomaly_types = set()
        for sample in self.samples:
            if sample['is_anomaly'] and 'anomaly_type' in sample:
                anomaly_types.add(sample['anomaly_type'])
        return list(anomaly_types)
