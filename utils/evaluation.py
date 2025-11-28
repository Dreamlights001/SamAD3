import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.ndimage import gaussian_filter

def compute_image_level_auroc(pred_scores, gt_labels):
    """
    计算图像级AUROC
    
    Args:
        pred_scores (list): 预测分数列表，每个元素是图像的异常分数
        gt_labels (list): 真实标签列表，0表示正常，1表示异常
    
    Returns:
        float: 图像级AUROC值
    """
    if len(set(gt_labels)) < 2:
        print("警告：图像级AUROC计算需要至少两个类别的样本")
        return 0.0
    
    return roc_auc_score(gt_labels, pred_scores)

def compute_pixel_level_auroc(pred_masks, gt_masks):
    """
    计算像素级AUROC
    
    Args:
        pred_masks (list): 预测掩码列表，每个元素是形状为(H, W)的numpy数组
        gt_masks (list): 真实掩码列表，每个元素是形状为(H, W)的numpy数组
    
    Returns:
        float: 像素级AUROC值
    """
    if not pred_masks or not gt_masks:
        print("警告：像素级AUROC计算需要预测掩码和真实掩码")
        return 0.0
    
    # 将所有掩码展平
    all_pred_pixels = []
    all_gt_pixels = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # 确保掩码形状一致
        if pred_mask.shape != gt_mask.shape:
            # 调整预测掩码大小以匹配真实掩码
            from skimage.transform import resize
            pred_mask = resize(pred_mask, gt_mask.shape, order=1, preserve_range=True)
        
        # 将掩码展平并添加到列表
        all_pred_pixels.extend(pred_mask.flatten())
        all_gt_pixels.extend(gt_mask.flatten())
    
    if len(set(all_gt_pixels)) < 2:
        print("警告：像素级AUROC计算需要至少两个类别的像素")
        return 0.0
    
    return roc_auc_score(all_gt_pixels, all_pred_pixels)

def compute_pixel_level_aupro(pred_masks, gt_masks, num_thresholds=100):
    """
    计算像素级AUPRO (Average Precision over Recall)
    这是异常检测中常用的指标，特别适合不平衡数据
    
    Args:
        pred_masks (list): 预测掩码列表，每个元素是形状为(H, W)的numpy数组
        gt_masks (list): 真实掩码列表，每个元素是形状为(H, W)的numpy数组
        num_thresholds (int): 阈值数量
    
    Returns:
        float: 像素级AUPRO值
    """
    if not pred_masks or not gt_masks:
        print("警告：像素级AUPRO计算需要预测掩码和真实掩码")
        return 0.0
    
    # 计算每个图像的前景区域（异常像素数）
    foreground_areas = []
    for gt_mask in gt_masks:
        foreground_area = np.sum(gt_mask > 0.5)
        if foreground_area > 0:  # 只考虑异常图像
            foreground_areas.append(foreground_area)
    
    if not foreground_areas:
        print("警告：没有异常图像用于计算AUPRO")
        return 0.0
    
    # 按前景区域排序
    sorted_indices = np.argsort(foreground_areas)
    sorted_foreground_areas = [foreground_areas[i] for i in sorted_indices]
    
    # 为每个阈值计算PR曲线
    thresholds = np.linspace(0, 1, num_thresholds + 1)
    precision_values = []
    recall_values = []
    
    for threshold in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i, idx in enumerate(sorted_indices):
            pred_mask = pred_masks[idx]
            gt_mask = gt_masks[idx]
            
            # 确保掩码形状一致
            if pred_mask.shape != gt_mask.shape:
                from skimage.transform import resize
                pred_mask = resize(pred_mask, gt_mask.shape, order=1, preserve_range=True)
            
            # 应用阈值
            binary_pred = (pred_mask >= threshold).astype(np.float32)
            
            # 计算TP, FP, FN
            tp = np.sum((binary_pred == 1) & (gt_mask > 0.5))
            fp = np.sum((binary_pred == 1) & (gt_mask <= 0.5))
            fn = np.sum((binary_pred == 0) & (gt_mask > 0.5))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # 计算精确率和召回率
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    # 计算AUPRO
    aupro = 0.0
    for i in range(len(thresholds) - 1):
        aupro += (recall_values[i+1] - recall_values[i]) * precision_values[i]
    
    return aupro

def compute_pixel_level_ap(pred_masks, gt_masks):
    """
    计算像素级平均精确率 (AP)
    
    Args:
        pred_masks (list): 预测掩码列表，每个元素是形状为(H, W)的numpy数组
        gt_masks (list): 真实掩码列表，每个元素是形状为(H, W)的numpy数组
    
    Returns:
        float: 像素级AP值
    """
    if not pred_masks or not gt_masks:
        print("警告：像素级AP计算需要预测掩码和真实掩码")
        return 0.0
    
    # 将所有掩码展平
    all_pred_pixels = []
    all_gt_pixels = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # 确保掩码形状一致
        if pred_mask.shape != gt_mask.shape:
            from skimage.transform import resize
            pred_mask = resize(pred_mask, gt_mask.shape, order=1, preserve_range=True)
        
        # 将掩码展平并添加到列表
        all_pred_pixels.extend(pred_mask.flatten())
        all_gt_pixels.extend(gt_mask.flatten())
    
    if len(set(all_gt_pixels)) < 2:
        print("警告：像素级AP计算需要至少两个类别的像素")
        return 0.0
    
    return average_precision_score(all_gt_pixels, all_pred_pixels)

def compute_f1_score(pred_masks, gt_masks, threshold=0.5):
    """
    计算F1分数
    
    Args:
        pred_masks (list): 预测掩码列表，每个元素是形状为(H, W)的numpy数组
        gt_masks (list): 真实掩码列表，每个元素是形状为(H, W)的numpy数组
        threshold (float): 阈值
    
    Returns:
        float: F1分数值
    """
    if not pred_masks or not gt_masks:
        print("警告：F1分数计算需要预测掩码和真实掩码")
        return 0.0
    
    all_binary_preds = []
    all_binary_gts = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # 确保掩码形状一致
        if pred_mask.shape != gt_mask.shape:
            from skimage.transform import resize
            pred_mask = resize(pred_mask, gt_mask.shape, order=1, preserve_range=True)
        
        # 应用阈值
        binary_pred = (pred_mask >= threshold).astype(np.int32)
        binary_gt = (gt_mask > 0.5).astype(np.int32)
        
        # 展平并添加到列表
        all_binary_preds.extend(binary_pred.flatten())
        all_binary_gts.extend(binary_gt.flatten())
    
    return f1_score(all_binary_gts, all_binary_preds)

def calculate_optimal_threshold(pred_scores, gt_labels):
    """
    计算最佳阈值（最大化F1分数）
    
    Args:
        pred_scores (list): 预测分数列表
        gt_labels (list): 真实标签列表
    
    Returns:
        float: 最佳阈值
    """
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        binary_preds = [1 if score >= threshold else 0 for score in pred_scores]
        f1 = f1_score(gt_labels, binary_preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def compute_segmentation_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    计算分割指标：IoU、Dice系数等
    
    Args:
        pred_masks (list): 预测掩码列表
        gt_masks (list): 真实掩码列表
        threshold (float): 阈值
    
    Returns:
        dict: 包含分割指标的字典
    """
    if not pred_masks or not gt_masks:
        print("警告：分割指标计算需要预测掩码和真实掩码")
        return {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # 确保掩码形状一致
        if pred_mask.shape != gt_mask.shape:
            from skimage.transform import resize
            pred_mask = resize(pred_mask, gt_mask.shape, order=1, preserve_range=True)
        
        # 应用阈值
        binary_pred = (pred_mask >= threshold).astype(np.float32)
        binary_gt = (gt_mask > 0.5).astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(binary_pred * binary_gt)
        union = np.sum(binary_pred + binary_gt) - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-8)
        
        # 计算Dice系数
        dice = 2 * intersection / (np.sum(binary_pred) + np.sum(binary_gt) + 1e-8)
        
        # 计算精确率和召回率
        precision = intersection / (np.sum(binary_pred) + 1e-8)
        recall = intersection / (np.sum(binary_gt) + 1e-8)
        
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
    
    num_samples = len(pred_masks)
    
    return {
        'iou': total_iou / num_samples,
        'dice': total_dice / num_samples,
        'precision': total_precision / num_samples,
        'recall': total_recall / num_samples
    }

def compute_evaluation_metrics(pred_masks=None, gt_masks=None, pred_scores=None, gt_labels=None, threshold=0.5):
    """
    计算所有评估指标
    
    Args:
        pred_masks (list, optional): 预测掩码列表
        gt_masks (list, optional): 真实掩码列表
        pred_scores (list, optional): 预测分数列表
        gt_labels (list, optional): 真实标签列表
        threshold (float, optional): 阈值
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    metrics = {}
    
    # 计算图像级指标
    if pred_scores is not None and gt_labels is not None:
        metrics['image_level_auroc'] = compute_image_level_auroc(pred_scores, gt_labels)
        
        # 计算最佳阈值和对应的F1分数
        if len(set(gt_labels)) >= 2:
            best_threshold = calculate_optimal_threshold(pred_scores, gt_labels)
            binary_preds = [1 if score >= best_threshold else 0 for score in pred_scores]
            metrics['f1_score'] = f1_score(gt_labels, binary_preds)
            metrics['best_threshold'] = best_threshold
    
    # 计算像素级指标
    if pred_masks is not None and gt_masks is not None:
        metrics['pixel_level_auroc'] = compute_pixel_level_auroc(pred_masks, gt_masks)
        metrics['pixel_level_ap'] = compute_pixel_level_ap(pred_masks, gt_masks)
        metrics['pixel_level_aupro'] = compute_pixel_level_aupro(pred_masks, gt_masks)
        metrics['f1_score'] = compute_f1_score(pred_masks, gt_masks, threshold)
        
        # 计算分割指标
        segmentation_metrics = compute_segmentation_metrics(pred_masks, gt_masks, threshold)
        metrics.update(segmentation_metrics)
    
    return metrics

def normalize_anomaly_scores(scores, method='minmax'):
    """
    归一化异常分数
    
    Args:
        scores (list or np.ndarray): 异常分数
        method (str): 归一化方法，支持 'minmax' 或 'zscore'
    
    Returns:
        np.ndarray: 归一化后的分数
    """
    scores = np.array(scores)
    
    if method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.zeros_like(scores)
    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 0:
            return (scores - mean_score) / std_score
        else:
            return np.zeros_like(scores)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

def apply_gaussian_smoothing(masks, sigma=1.0):
    """
    对掩码应用高斯平滑，减少噪声
    
    Args:
        masks (list): 掩码列表
        sigma (float): 高斯核标准差
    
    Returns:
        list: 平滑后的掩码列表
    """
    smoothed_masks = []
    
    for mask in masks:
        smoothed_mask = gaussian_filter(mask, sigma=sigma)
        # 保持值在0-1范围内
        smoothed_mask = np.clip(smoothed_mask, 0, 1)
        smoothed_masks.append(smoothed_mask)
    
    return smoothed_masks

def compute_per_class_metrics(pred_masks_by_class, gt_masks_by_class, anomaly_types):
    """
    计算每个异常类型的评估指标
    
    Args:
        pred_masks_by_class (dict): 按异常类型组织的预测掩码
        gt_masks_by_class (dict): 按异常类型组织的真实掩码
        anomaly_types (list): 异常类型列表
    
    Returns:
        dict: 每个异常类型的评估指标
    """
    class_metrics = {}
    
    for anomaly_type in anomaly_types:
        if anomaly_type in pred_masks_by_class and anomaly_type in gt_masks_by_class:
            metrics = compute_evaluation_metrics(
                pred_masks=pred_masks_by_class[anomaly_type],
                gt_masks=gt_masks_by_class[anomaly_type]
            )
            class_metrics[anomaly_type] = metrics
    
    return class_metrics
