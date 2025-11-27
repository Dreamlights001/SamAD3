# 数据预处理管道
from data_loaders.dataset_loader import get_dataset_loader
from utils.image_processing import preprocess_image
from utils.data_augmentation import augment_batch

class DataPipeline:
    """数据预处理管道"""
    def __init__(self, dataset_name, category, use_augmentation=False, 
                 num_augmented_per_image=1):
        """初始化数据预处理管道"""
        # 获取数据集加载器
        self.dataset_loader = get_dataset_loader(dataset_name, category)
        self.use_augmentation = use_augmentation
        self.num_augmented_per_image = num_augmented_per_image
    
    def get_test_data(self):
        """获取测试数据"""
        # 获取正常图像
        normal_images = self.dataset_loader.get_normal_images()
        
        # 获取异常图像和掩码
        anomaly_data = self.dataset_loader.get_anomaly_images_and_masks()
        
        # 预处理正常图像
        processed_normal_images = []
        for image in normal_images:
            processed_image = preprocess_image(image)
            processed_normal_images.append(processed_image)
        
        # 预处理异常图像和掩码
        processed_anomaly_data = []
        for image, mask, category in anomaly_data:
            processed_image = preprocess_image(image)
            processed_anomaly_data.append((processed_image, mask, category))
        
        return processed_normal_images, processed_anomaly_data
    
    def get_train_data(self):
        """获取训练数据（用于特征提取或微调）"""
        # 获取正常图像作为训练数据
        normal_images = self.dataset_loader.get_normal_images()
        
        # 预处理图像
        processed_images = []
        for image in normal_images:
            processed_image = preprocess_image(image)
            processed_images.append(processed_image)
        
        # 应用数据增强
        if self.use_augmentation:
            processed_images = augment_batch(
                processed_images, 
                self.num_augmented_per_image
            )
        
        return processed_images
    
    def iterate_test_data(self):
        """迭代测试数据"""
        normal_images, anomaly_data = self.get_test_data()
        
        # 先迭代正常图像
        for image in normal_images:
            yield image, None, "normal"
        
        # 再迭代异常图像
        for image, mask, category in anomaly_data:
            yield image, mask, category

def create_data_pipeline(dataset_name, category, **kwargs):
    """创建数据预处理管道的工厂函数"""
    return DataPipeline(dataset_name, category, **kwargs)
