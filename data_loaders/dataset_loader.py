# 数据集加载模块
import os
import glob
from PIL import Image
import numpy as np
from config import DATASET_ROOT, supported_datasets

class DatasetLoader:
    """数据集加载器基类"""
    def __init__(self, dataset_name, category=None):
        """初始化数据集加载器"""
        if dataset_name not in supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {supported_datasets}")
        
        self.dataset_name = dataset_name
        self.category = category
        self.dataset_path = os.path.join(DATASET_ROOT, dataset_name)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
    
    def load_image(self, image_path):
        """加载单张图像"""
        return Image.open(image_path).convert("RGB")
    
    def load_mask(self, mask_path):
        """加载单张掩码"""
        mask = Image.open(mask_path).convert("L")
        return np.array(mask) > 0  # 转换为二值掩码
    
    def get_normal_images(self):
        """获取正常图像列表"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_anomaly_images_and_masks(self):
        """获取异常图像和对应的掩码列表"""
        raise NotImplementedError("子类必须实现此方法")

class MVTecDatasetLoader(DatasetLoader):
    """MVTec数据集加载器"""
    def __init__(self, category):
        super().__init__("mvtec", category)
        self.category_path = os.path.join(self.dataset_path, category)
    
    def get_normal_images(self):
        """获取MVTec数据集中的正常图像"""
        normal_image_dir = os.path.join(self.category_path, "test", "good")
        image_paths = glob.glob(os.path.join(normal_image_dir, "*.png")) + \
                     glob.glob(os.path.join(normal_image_dir, "*.jpg"))
        return [self.load_image(path) for path in image_paths]
    
    def get_anomaly_images_and_masks(self):
        """获取MVTec数据集中的异常图像和掩码"""
        anomaly_categories = [d for d in os.listdir(os.path.join(self.category_path, "test")) \
                              if d != "good"]
        
        image_mask_pairs = []
        for category in anomaly_categories:
            # 获取异常图像
            image_dir = os.path.join(self.category_path, "test", category)
            image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                         glob.glob(os.path.join(image_dir, "*.jpg"))
            
            # 获取对应掩码
            mask_dir = os.path.join(self.category_path, "ground_truth", category)
            mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
            
            # 配对图像和掩码
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                mask_name = img_name.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
                mask_path = os.path.join(mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    image = self.load_image(img_path)
                    mask = self.load_mask(mask_path)
                    image_mask_pairs.append((image, mask, category))
        
        return image_mask_pairs

class VisADatasetLoader(DatasetLoader):
    """VisA数据集加载器"""
    def __init__(self, category):
        super().__init__("visa", category)
        self.category_path = os.path.join(self.dataset_path, category, "Data")
    
    def get_normal_images(self):
        """获取VisA数据集中的正常图像"""
        normal_image_dir = os.path.join(self.category_path, "Images", "Normal")
        image_paths = glob.glob(os.path.join(normal_image_dir, "*.png")) + \
                     glob.glob(os.path.join(normal_image_dir, "*.jpg"))
        return [self.load_image(path) for path in image_paths]
    
    def get_anomaly_images_and_masks(self):
        """获取VisA数据集中的异常图像和掩码"""
        # 获取异常图像
        anomaly_image_dir = os.path.join(self.category_path, "Images", "Anomaly")
        image_paths = glob.glob(os.path.join(anomaly_image_dir, "*.png")) + \
                     glob.glob(os.path.join(anomaly_image_dir, "*.jpg"))
        
        # 获取对应掩码
        mask_dir = os.path.join(self.category_path, "Masks", "Anomaly")
        
        image_mask_pairs = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, img_name)
            
            if os.path.exists(mask_path):
                image = self.load_image(img_path)
                mask = self.load_mask(mask_path)
                image_mask_pairs.append((image, mask, "anomaly"))
        
        return image_mask_pairs

def get_dataset_loader(dataset_name, category=None):
    """工厂函数，返回对应的数据集加载器"""
    if dataset_name == "mvtec":
        return MVTecDatasetLoader(category)
    elif dataset_name == "visa":
        return VisADatasetLoader(category)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
