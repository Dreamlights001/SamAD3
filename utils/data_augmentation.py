# 数据增强模块
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random

def random_rotation(image, max_angle=10):
    """随机旋转图像"""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, fillcolor=(0, 0, 0))

def random_flip(image):
    """随机翻转图像"""
    # 随机决定是否水平翻转
    if random.random() > 0.5:
        image = ImageOps.mirror(image)
    # 随机决定是否垂直翻转
    if random.random() > 0.5:
        image = ImageOps.flip(image)
    return image

def random_brightness(image, min_factor=0.8, max_factor=1.2):
    """随机调整亮度"""
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)

def random_contrast(image, min_factor=0.8, max_factor=1.2):
    """随机调整对比度"""
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)

def random_crop(image, crop_ratio=0.9):
    """随机裁剪图像"""
    width, height = image.size
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    # 随机选择裁剪区域
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    # 裁剪并调整回原始大小
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height))

def augment_image(image, apply_rotation=True, apply_flip=True, 
                  apply_brightness=True, apply_contrast=True, apply_crop=True):
    """综合数据增强"""
    augmented_image = image.copy()
    
    # 随机裁剪
    if apply_crop and random.random() > 0.5:
        augmented_image = random_crop(augmented_image)
    
    # 随机旋转
    if apply_rotation and random.random() > 0.5:
        augmented_image = random_rotation(augmented_image)
    
    # 随机翻转
    if apply_flip and random.random() > 0.5:
        augmented_image = random_flip(augmented_image)
    
    # 随机调整亮度
    if apply_brightness and random.random() > 0.5:
        augmented_image = random_brightness(augmented_image)
    
    # 随机调整对比度
    if apply_contrast and random.random() > 0.5:
        augmented_image = random_contrast(augmented_image)
    
    return augmented_image

def augment_batch(images, num_augmented_per_image=1):
    """批量增强图像"""
    augmented_images = []
    
    for image in images:
        # 添加原始图像
        augmented_images.append(image)
        
        # 添加增强后的图像
        for _ in range(num_augmented_per_image):
            augmented = augment_image(image)
            augmented_images.append(augmented)
    
    return augmented_images
