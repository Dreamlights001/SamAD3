import os
import glob
from typing import List, Dict, Optional
from .base_dataset import AnomalyDataset

class VisADataset(AnomalyDataset):
    """
    VisA异常检测数据集加载器
    支持VisA数据集格式
    """
    
    def __init__(self,
                 root_dir: str,
                 category: str,
                 split: str = 'test',
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 load_masks: bool = True):
        """
        初始化VisA数据集
        
        Args:
            root_dir: 数据集根目录，例如'~/autodl-tmp/datasets/visa'
            category: 数据集类别，如'candle', 'capsules'等
            split: 数据集分割（'train'或'test'）
            transform: 图像变换函数
            target_transform: 目标变换函数
            load_masks: 是否加载掩码
        """
        super().__init__(
            root_dir=root_dir,
            dataset_name='visa',
            category=category,
            split=split,
            transform=transform,
            target_transform=target_transform,
            load_masks=load_masks
        )
    
    def _load_samples(self):
        """
        加载VisA数据集的样本信息
        VisA格式：root/category/Data/Images/{Anomaly,Normal}/
        掩码格式：root/category/Data/Masks/Anomaly/
        """
        # 构建类别目录路径
        category_data_dir = os.path.join(self.root_dir, self.category, 'Data')
        
        # 确保目录存在
        if not os.path.isdir(category_data_dir):
            raise FileNotFoundError(f"类别数据目录不存在: {category_data_dir}")
        
        # VisA数据集没有明确的train/test分割，我们根据异常/正常来区分
        # 但保持与基类一致的接口，通过split参数可以过滤
        
        # 加载正常样本
        normal_images_dir = os.path.join(category_data_dir, 'Images', 'Normal')
        if os.path.isdir(normal_images_dir):
            normal_image_paths = glob.glob(os.path.join(normal_images_dir, '*.png')) + \
                               glob.glob(os.path.join(normal_images_dir, '*.jpg')) + \
                               glob.glob(os.path.join(normal_images_dir, '*.jpeg'))
            normal_image_paths.sort()
            
            for image_path in normal_image_paths:
                sample_info = {
                    'image_path': image_path,
                    'is_anomaly': False,
                    'anomaly_type': 'normal'
                }
                self.samples.append(sample_info)
        
        # 加载异常样本
        anomaly_images_dir = os.path.join(category_data_dir, 'Images', 'Anomaly')
        if os.path.isdir(anomaly_images_dir):
            anomaly_image_paths = glob.glob(os.path.join(anomaly_images_dir, '*.png')) + \
                                 glob.glob(os.path.join(anomaly_images_dir, '*.jpg')) + \
                                 glob.glob(os.path.join(anomaly_images_dir, '*.jpeg'))
            anomaly_image_paths.sort()
            
            for image_path in anomaly_image_paths:
                sample_info = {
                    'image_path': image_path,
                    'is_anomaly': True,
                    'anomaly_type': 'anomaly'  # VisA没有细分异常类型
                }
                
                # 如果需要掩码，找到对应的掩码路径
                if self.load_masks:
                    mask_dir = os.path.join(category_data_dir, 'Masks', 'Anomaly')
                    image_name = os.path.basename(image_path)
                    mask_path = os.path.join(mask_dir, image_name)
                    
                    if os.path.exists(mask_path):
                        sample_info['mask_path'] = mask_path
                    else:
                        sample_info['mask_path'] = None
                
                self.samples.append(sample_info)
        
        # 如果指定了split，进行过滤
        # 注意：VisA数据集没有明确的train/test分割，这里我们做一个简单的划分
        # 实际使用中可能需要根据split_csv文件来进行划分
        if self.split == 'train':
            # 对于训练集，我们只使用正常样本
            self.samples = [s for s in self.samples if not s['is_anomaly']]
        elif self.split == 'test':
            # 对于测试集，使用所有样本（正常和异常）
            pass
    
    def get_category_names(self) -> List[str]:
        """
        获取VisA数据集中的所有类别名称
        """
        # 列出根目录下的所有子目录
        categories = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d)) and 
                     not d.startswith('.') and 
                     os.path.isdir(os.path.join(self.root_dir, d, 'Data'))]
        return categories

# 创建一个工厂函数，用于获取VisA数据集
def get_visa_dataset(
    root_dir: str,
    category: str,
    split: str = 'test',
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None,
    load_masks: bool = True
) -> VisADataset:
    """
    获取VisA数据集实例
    
    Args:
        root_dir: 数据集根目录
        category: 数据集类别
        split: 数据集分割
        transform: 图像变换函数
        target_transform: 目标变换函数
        load_masks: 是否加载掩码
        
    Returns:
        VisA数据集实例
    """
    return VisADataset(
        root_dir=root_dir,
        category=category,
        split=split,
        transform=transform,
        target_transform=target_transform,
        load_masks=load_masks
    )
