# 图像预处理模块
import numpy as np
from PIL import Image

def resize_image(image, max_size=1024):
    """调整图像大小，保持纵横比"""
    width, height = image.size
    
    # 计算缩放比例
    if width > height:
        ratio = max_size / width
        new_width = max_size
        new_height = int(height * ratio)
    else:
        ratio = max_size / height
        new_width = int(width * ratio)
        new_height = max_size
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def normalize_image(image):
    """将图像标准化到[0, 1]范围"""
    img_array = np.array(image)
    return img_array / 255.0

def pad_image(image, target_size):
    """对图像进行填充，使其达到目标大小"""
    width, height = image.size
    target_width, target_height = target_size
    
    # 计算填充量
    pad_width = max(0, target_width - width) // 2
    pad_height = max(0, target_height - height) // 2
    
    # 创建新图像
    new_image = Image.new(image.mode, (target_width, target_height), color=(0, 0, 0))
    
    # 将原图粘贴到中心
    new_image.paste(image, (pad_width, pad_height))
    
    return new_image

def preprocess_image(image, max_size=1024):
    """图像预处理流程"""
    # 调整大小
    resized_image = resize_image(image, max_size)
    
    # 转换为RGB格式
    rgb_image = resized_image.convert("RGB")
    
    return rgb_image

def postprocess_mask(mask, original_size):
    """对预测掩码进行后处理，恢复到原始图像大小"""
    # 将掩码转换为PIL图像
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    
    # 调整回原始大小
    resized_mask = mask_pil.resize(original_size, Image.NEAREST)
    
    # 转换回布尔掩码
    return np.array(resized_mask) > 127

def visualize_prediction(image, pred_mask, gt_mask=None, alpha=0.5):
    """
    可视化预测结果
    
    参数:
        image: 原始图像 (numpy array 或 PIL Image)
        pred_mask: 预测掩码 (numpy array)
        gt_mask: 真实掩码 (numpy array, 可选)
        alpha: 掩码叠加的透明度
    
    返回:
        visualization: 可视化结果图像 (numpy array)
    """
    import cv2
    
    # 确保图像是numpy数组
    if hasattr(image, 'numpy'):
        image = image.numpy()
    elif hasattr(image, 'toarray'):
        image = image.toarray()
    elif hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 确保图像范围是0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 创建可视化图像的副本
    visualization = image.copy()
    
    # 调整掩码大小以匹配图像
    h, w = image.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 在预测掩码区域上绘制半透明红色
    if pred_mask.any():
        # 创建红色掩码
        red_mask = np.zeros_like(visualization)
        red_mask[pred_mask > 0] = [255, 0, 0]  # BGR格式的红色
        
        # 叠加掩码，设置透明度
        visualization = cv2.addWeighted(visualization, 1 - alpha, red_mask, alpha, 0)
    
    # 如果提供了真实掩码，在上面绘制半透明绿色
    if gt_mask is not None:
        # 调整真实掩码大小
        if gt_mask.shape[:2] != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if gt_mask.any():
            # 创建绿色掩码
            green_mask = np.zeros_like(visualization)
            green_mask[gt_mask > 0] = [0, 255, 0]  # BGR格式的绿色
            
            # 叠加掩码，设置透明度
            visualization = cv2.addWeighted(visualization, 0.9, green_mask, 0.2, 0)
    
    return visualization


def visualize_heatmap(image, mask, colormap=cv2.COLORMAP_JET):
    """
    使用热力图可视化掩码
    
    参数:
        image: 原始图像 (numpy array 或 PIL Image)
        mask: 掩码 (numpy array)
        colormap: OpenCV颜色映射
    
    返回:
        heatmap_overlay: 热力图叠加结果 (numpy array)
    """
    import cv2
    
    # 确保图像是numpy数组
    if hasattr(image, 'numpy'):
        image = image.numpy()
    elif hasattr(image, 'toarray'):
        image = image.toarray()
    elif hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 确保图像范围是0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 调整掩码大小以匹配图像
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 归一化掩码到0-255范围
    mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(mask_normalized, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为RGB
    
    # 叠加热力图到原始图像
    heatmap_overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    return heatmap_overlay


def create_visualization_grid(images, titles=None, figsize=(15, 10)):
    """
    创建可视化网格显示多张图像
    
    参数:
        images: 图像列表 (numpy arrays)
        titles: 图像标题列表 (可选)
        figsize: 图表大小
    
    返回:
        fig: matplotlib图表对象
    """
    import matplotlib.pyplot as plt
    
    # 计算网格尺寸
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    # 创建图表
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 显示图像
    for i, img in enumerate(images):
        # 确保图像是RGB格式
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            # OpenCV图像是BGR格式，转换为RGB
            if img.shape[2] == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img)
        
        # 添加标题
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        
        # 隐藏坐标轴
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_comparison(image, pred_mask, gt_mask=None, metrics=None):
    """
    创建完整的预测结果比较可视化
    
    参数:
        image: 原始图像 (numpy array 或 PIL Image)
        pred_mask: 预测掩码 (numpy array)
        gt_mask: 真实掩码 (numpy array, 可选)
        metrics: 评估指标字典 (可选)
    
    返回:
        fig: matplotlib图表对象
    """
    import matplotlib.pyplot as plt
    
    # 确保图像是numpy数组
    if hasattr(image, 'numpy'):
        image = image.numpy()
    elif hasattr(image, 'toarray'):
        image = image.toarray()
    elif hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    # 准备可视化图像
    images = [image]
    titles = ['原始图像']
    
    # 添加预测掩码可视化
    pred_visualization = visualize_prediction(image, pred_mask)
    images.append(pred_visualization)
    titles.append('预测结果')
    
    # 添加预测热力图
    pred_heatmap = visualize_heatmap(image, pred_mask)
    images.append(pred_heatmap)
    titles.append('预测热力图')
    
    # 如果有真实掩码，添加额外的可视化
    if gt_mask is not None:
        # 添加真实掩码可视化
        gt_visualization = visualize_prediction(image, np.zeros_like(pred_mask), gt_mask)
        images.append(gt_visualization)
        titles.append('真实掩码')
        
        # 添加对比可视化
        comparison = visualize_prediction(image, pred_mask, gt_mask)
        images.append(comparison)
        titles.append('预测与真实对比')
        
        # 添加真实掩码热力图
        gt_heatmap = visualize_heatmap(image, gt_mask)
        images.append(gt_heatmap)
        titles.append('真实掩码热力图')
    
    # 创建可视化网格
    fig = create_visualization_grid(images, titles)
    
    # 添加指标文本
    if metrics:
        metrics_text = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))])
        fig.suptitle(f'评估指标: {metrics_text}', fontsize=14, y=0.98)
    
    return fig


def save_visualization(image, save_path, dpi=300):
    """
    保存可视化结果
    
    参数:
        image: matplotlib图表对象或numpy数组
        save_path: 保存路径
        dpi: 图像DPI
    """
    import matplotlib.pyplot as plt
    import cv2
    import os
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    if hasattr(image, 'savefig'):  # matplotlib图表
        plt.figure(image.number)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(image)
    else:  # numpy数组
        # 如果是RGB格式，转换为BGR用于OpenCV保存
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
    
    print(f'可视化结果已保存至: {save_path}')
