# 评估指标计算模块
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.measure import label

def calculate_iou(pred_mask, true_mask):
    """计算交并比(IoU)"""
    # 确保输入是布尔数组
    pred_mask = np.array(pred_mask).astype(bool)
    true_mask = np.array(true_mask).astype(bool)
    
    # 计算交集和并集
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    # 避免除零错误
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_precision(pred_mask, true_mask):
    """计算精确率"""
    pred_mask = np.array(pred_mask).astype(bool)
    true_mask = np.array(true_mask).astype(bool)
    
    # 计算真正例和预测为正例的数量
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    
    # 避免除零错误
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)

def calculate_recall(pred_mask, true_mask):
    """计算召回率"""
    pred_mask = np.array(pred_mask).astype(bool)
    true_mask = np.array(true_mask).astype(bool)
    
    # 计算真正例和实际正例的数量
    tp = np.logical_and(pred_mask, true_mask).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    
    # 避免除零错误
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)

def calculate_f1_score(precision, recall):
    """计算F1分数"""
    # 避免除零错误
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_image_level_auroc(pred_masks, true_masks):
    """计算图像级AUROC"""
    # 将每张图像的掩码转换为二进制标签（有缺陷=1，无缺陷=0）
    y_true = [1 if np.any(mask) else 0 for mask in true_masks]
    y_score = [np.any(mask) for mask in pred_masks]
    
    # 计算AUROC
    if len(set(y_true)) > 1:  # 确保正负样本都存在
        return roc_auc_score(y_true, y_score)
    return 0.5  # 如果只有一类样本，返回0.5

def calculate_pixel_level_auroc(pred_masks, true_masks):
    """计算像素级AUROC"""
    # 展平所有掩码
    y_true = np.concatenate([mask.flatten() for mask in true_masks]).astype(int)
    y_score = np.concatenate([mask.flatten() for mask in pred_masks])
    
    # 计算AUROC
    if len(set(y_true)) > 1:  # 确保正负样本都存在
        return roc_auc_score(y_true, y_score)
    return 0.5

def calculate_ap(pred_masks, true_masks):
    """计算平均精度(AP)"""
    # 展平所有掩码
    y_true = np.concatenate([mask.flatten() for mask in true_masks]).astype(int)
    y_score = np.concatenate([mask.flatten() for mask in pred_masks])
    
    # 计算AP
    if len(set(y_true)) > 1:  # 确保正负样本都存在
        return average_precision_score(y_true, y_score)
    return 0.0

def calculate_aupro(pred_masks, true_masks):
    """计算AUPRO（PR曲线下面积，按区域组织）"""
    # 实现基于区域的PR曲线下面积计算
    pr_values = []
    
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        # 标记连通区域
        labeled_true, num_true_regions = label(true_mask, return_num=True)
        
        if num_true_regions == 0:
            continue
        
        for region_id in range(1, num_true_regions + 1):
            # 获取当前真实区域
            region_mask = labeled_true == region_id
            
            # 计算该区域的精确率和召回率
            pred_region = np.logical_and(pred_mask, region_mask)
            true_positives = pred_region.sum()
            
            # 精确率：区域内正确预测的像素数 / 区域内总预测像素数
            region_pred_pixels = np.logical_and(pred_mask, region_mask).sum()
            precision = true_positives / region_pred_pixels if region_pred_pixels > 0 else 0
            
            # 召回率：区域内正确预测的像素数 / 区域内真实像素数
            region_true_pixels = region_mask.sum()
            recall = true_positives / region_true_pixels if region_true_pixels > 0 else 0
            
            pr_values.append((precision, recall))
    
    if not pr_values:
        return 0.0
    
    # 计算平均PR值作为简化的AUPRO
    avg_precision = np.mean([p for p, r in pr_values])
    return avg_precision

def evaluate_prediction(pred_mask, true_mask):
    """评估预测结果"""
    # 计算各项指标
    iou = calculate_iou(pred_mask, true_mask)
    precision = calculate_precision(pred_mask, true_mask)
    recall = calculate_recall(pred_mask, true_mask)
    f1_score = calculate_f1_score(precision, recall)
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def evaluate_batch_predictions(pred_masks, true_masks):
    """评估批量预测结果，计算所有指标"""
    # 计算传统指标的平均值
    ious = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        metrics = evaluate_prediction(pred_mask, true_mask)
        ious.append(metrics['iou'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
    
    # 计算高级评估指标
    image_level_auroc = calculate_image_level_auroc(pred_masks, true_masks)
    pixel_level_auroc = calculate_pixel_level_auroc(pred_masks, true_masks)
    ap = calculate_ap(pred_masks, true_masks)
    aupro = calculate_aupro(pred_masks, true_masks)
    
    return {
        'average_iou': np.mean(ious),
        'average_precision': np.mean(precisions),
        'average_recall': np.mean(recalls),
        'average_f1_score': np.mean(f1_scores),
        'image_level_auroc': image_level_auroc,
        'pixel_level_auroc': pixel_level_auroc,
        'ap': ap,
        'aupro': aupro
    }

# visualize_prediction函数已移至utils/image_processing.py
