import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class ZeroShotAnomalyDetector:
    """
    零样本异常检测器，处理跨数据集推理和混合提示生成
    """
    
    def __init__(self, prompt_generator, detector):
        """
        初始化零样本异常检测器
        
        Args:
            prompt_generator: 提示词生成器实例
            detector: SAM3异常检测器实例
        """
        self.prompt_generator = prompt_generator
        self.detector = detector
        
        # 数据集映射表，用于跨数据集推理
        self.dataset_mappings = self._load_dataset_mappings()
        
        # 类别相似性映射
        self.category_similarities = self._load_category_similarities()
    
    def _load_dataset_mappings(self) -> Dict[str, List[str]]:
        """
        加载数据集映射表，定义数据集之间的对应关系
        
        Returns:
            dict: 数据集映射关系
        """
        # 定义数据集之间的映射关系
        mappings = {
            'mvtec': {
                'visa': {
                    # MVTec 到 VisA 的类别映射
                    'bottle': 'bottle',
                    'cable': 'cable',
                    'capsule': 'capsule',
                    'carpet': None,
                    'grid': None,
                    'hazelnut': 'hazelnut',
                    'leather': None,
                    'metal_nut': 'metal_nut',
                    'pill': 'pill',
                    'screw': 'screw',
                    'tile': None,
                    'toothbrush': None,
                    'transistor': None,
                    'wood': None,
                    'zipper': 'zipper'
                }
            },
            'visa': {
                'mvtec': {
                    # VisA 到 MVTec 的类别映射
                    'bottle': 'bottle',
                    'cable': 'cable',
                    'capsule': 'capsule',
                    'hazelnut': 'hazelnut',
                    'metal_nut': 'metal_nut',
                    'pill': 'pill',
                    'screw': 'screw',
                    'zipper': 'zipper'
                }
            }
        }
        
        return mappings
    
    def _load_category_similarities(self) -> Dict[str, Dict[str, float]]:
        """
        加载类别相似性映射，定义不同类别之间的相似程度
        
        Returns:
            dict: 类别相似性映射
        """
        # 预定义的类别相似性（0-1之间的值，表示相似程度）
        similarities = {
            'bottle': {
                'bottle': 1.0,
                'capsule': 0.6,
                'pill': 0.5
            },
            'cable': {
                'cable': 1.0,
                'screw': 0.4,
                'zipper': 0.3
            },
            'capsule': {
                'capsule': 1.0,
                'pill': 0.7,
                'bottle': 0.6
            },
            'pill': {
                'pill': 1.0,
                'capsule': 0.7,
                'bottle': 0.5
            },
            'screw': {
                'screw': 1.0,
                'metal_nut': 0.6,
                'cable': 0.4
            },
            'metal_nut': {
                'metal_nut': 1.0,
                'screw': 0.6
            },
            'zipper': {
                'zipper': 1.0,
                'cable': 0.3
            },
            'hazelnut': {
                'hazelnut': 1.0,
                'pill': 0.4
            }
        }
        
        return similarities
    
    def get_mapped_category(self, source_dataset: str, target_dataset: str, source_category: str) -> Optional[str]:
        """
        获取源数据集类别在目标数据集中的对应类别
        
        Args:
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            source_category: 源数据集类别
        
        Returns:
            str or None: 对应的目标数据集类别，如果没有映射则返回None
        """
        if source_dataset in self.dataset_mappings and target_dataset in self.dataset_mappings[source_dataset]:
            return self.dataset_mappings[source_dataset][target_dataset].get(source_category)
        return None
    
    def get_similar_categories(self, category: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        获取与指定类别相似的其他类别
        
        Args:
            category: 类别名称
            top_k: 返回的相似类别数量
        
        Returns:
            list: 相似类别及其相似度的列表
        """
        if category in self.category_similarities:
            similarities = self.category_similarities[category]
            # 按相似度排序并返回前top_k个
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_similarities[:top_k]
        return []
    
    def generate_cross_dataset_prompts(self, source_dataset: str, target_dataset: str, 
                                      source_category: str, target_category: str) -> List[str]:
        """
        生成跨数据集推理的提示词
        
        Args:
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            source_category: 源数据集类别
            target_category: 目标数据集类别
        
        Returns:
            list: 跨数据集推理的提示词列表
        """
        prompts = []
        
        # 1. 获取源数据集类别的提示词
        source_prompts = self.prompt_generator.get_dataset_prompts(source_dataset, source_category)
        
        # 2. 获取目标数据集类别的提示词
        target_prompts = self.prompt_generator.get_dataset_prompts(target_dataset, target_category)
        
        # 3. 生成混合提示词
        mixed_prompts = self.prompt_generator.generate_mixed_prompts(
            base_prompts=source_prompts,
            context_info=f"similar to {target_category}",
            dataset_name=source_dataset,
            category=source_category
        )
        
        # 4. 添加通用异常提示词
        general_prompts = self.prompt_generator.get_general_anomaly_prompts()
        
        # 合并所有提示词并去重
        prompts.extend(source_prompts)
        prompts.extend(target_prompts)
        prompts.extend(mixed_prompts)
        prompts.extend(general_prompts)
        
        return list(set(prompts))  # 去重
    
    def detect_anomalies_cross_dataset(self, image, source_dataset: str, target_dataset: str,
                                      source_category: str, target_category: str, 
                                      threshold: float = 0.5) -> Dict:
        """
        执行跨数据集异常检测
        
        Args:
            image: 输入图像
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            source_category: 源数据集类别
            target_category: 目标数据集类别
            threshold: 异常掩码阈值
        
        Returns:
            dict: 异常检测结果
        """
        # 生成跨数据集提示词
        prompts = self.generate_cross_dataset_prompts(
            source_dataset, target_dataset, source_category, target_category
        )
        
        # 执行异常检测
        results = self.detector.detect_anomalies(
            image=image,
            text_prompts=prompts,
            threshold=threshold
        )
        
        # 添加跨数据集推理的元数据
        results['cross_dataset_info'] = {
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
            'source_category': source_category,
            'target_category': target_category,
            'prompts_used': len(prompts)
        }
        
        return results
    
    def detect_anomalies_zeroshot(self, image, dataset_name: str, category: str,
                                 threshold: float = 0.5, use_similar_categories: bool = True) -> Dict:
        """
        执行零样本异常检测
        
        Args:
            image: 输入图像
            dataset_name: 目标数据集名称
            category: 目标数据集类别
            threshold: 异常掩码阈值
            use_similar_categories: 是否使用相似类别的提示词
        
        Returns:
            dict: 零样本异常检测结果
        """
        all_prompts = []
        
        # 1. 获取通用异常提示词
        general_prompts = self.prompt_generator.get_general_anomaly_prompts()
        all_prompts.extend(general_prompts)
        
        # 2. 获取数据集特定提示词（如果有）
        dataset_prompts = self.prompt_generator.get_dataset_prompts(dataset_name, category)
        all_prompts.extend(dataset_prompts)
        
        # 3. 如果启用，添加相似类别的提示词
        if use_similar_categories:
            similar_categories = self.get_similar_categories(category)
            for sim_category, similarity in similar_categories:
                # 只有当相似度足够高时才添加
                if similarity > 0.5:
                    for other_dataset in ['mvtec', 'visa']:
                        if other_dataset != dataset_name:
                            similar_prompts = self.prompt_generator.get_dataset_prompts(
                                other_dataset, sim_category
                            )
                            all_prompts.extend(similar_prompts)
        
        # 4. 生成混合提示词
        mixed_prompts = self.prompt_generator.generate_mixed_prompts(
            base_prompts=dataset_prompts,
            context_info=f"zero-shot anomaly detection",
            dataset_name=dataset_name,
            category=category
        )
        all_prompts.extend(mixed_prompts)
        
        # 去重提示词
        all_prompts = list(set(all_prompts))
        
        # 执行异常检测
        results = self.detector.detect_anomalies(
            image=image,
            text_prompts=all_prompts,
            threshold=threshold
        )
        
        # 添加零样本推理的元数据
        results['zeroshot_info'] = {
            'dataset': dataset_name,
            'category': category,
            'prompts_used': len(all_prompts),
            'used_similar_categories': use_similar_categories
        }
        
        return results
    
    def optimize_threshold(self, val_dataset, source_dataset: str, target_dataset: str,
                          source_category: str, target_category: str, 
                          threshold_range: Tuple[float, float] = (0.1, 0.9), 
                          num_thresholds: int = 9) -> float:
        """
        优化跨数据集推理的阈值
        
        Args:
            val_dataset: 验证数据集
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            source_category: 源数据集类别
            target_category: 目标数据集类别
            threshold_range: 阈值范围
            num_thresholds: 阈值数量
        
        Returns:
            float: 最佳阈值
        """
        from sklearn.metrics import f1_score
        import torchvision.transforms as transforms
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        best_f1 = 0.0
        best_threshold = 0.5
        
        all_pred_masks = []
        all_gt_masks = []
        
        # 收集预测结果和真实标签
        for sample in val_dataset:
            # 将tensor转换回PIL图像
            img_tensor = sample['image']
            img_pil = transforms.ToPILImage()(img_tensor)
            
            # 执行跨数据集异常检测
            results = self.detect_anomalies_cross_dataset(
                image=img_pil,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                source_category=source_category,
                target_category=target_category,
                threshold=0.5  # 使用中间值，后续将根据阈值重新计算
            )
            
            # 存储预测结果和真实标签
            if len(results['masks']) > 0:
                best_mask_idx = results['scores'].argmax()
                pred_mask = results['masks'][best_mask_idx].cpu().numpy()
            else:
                pred_mask = np.zeros((512, 512))
            
            all_pred_masks.append(pred_mask)
            all_gt_masks.append(sample['mask'].numpy())
        
        # 对每个阈值计算F1分数
        for threshold in thresholds:
            # 应用阈值
            binary_preds = []
            binary_gts = []
            
            for pred_mask, gt_mask in zip(all_pred_masks, all_gt_masks):
                binary_pred = (pred_mask >= threshold).astype(np.int32).flatten()
                binary_gt = (gt_mask > 0.5).astype(np.int32).flatten()
                
                binary_preds.extend(binary_pred)
                binary_gts.extend(binary_gt)
            
            # 计算F1分数
            current_f1 = f1_score(binary_gts, binary_preds)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        print(f"最佳阈值: {best_threshold:.3f}, F1分数: {best_f1:.4f}")
        return best_threshold
    
    def save_zeroshot_results(self, results: Dict, output_path: str):
        """
        保存零样本推理结果
        
        Args:
            results: 推理结果
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换numpy数组为可序列化的格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def load_zeroshot_results(self, results_path: str) -> Dict:
        """
        加载零样本推理结果
        
        Args:
            results_path: 结果文件路径
        
        Returns:
            dict: 加载的推理结果
        """
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # 转换列表回numpy数组
        for key, value in results.items():
            if isinstance(value, list):
                results[key] = np.array(value)
        
        return results


def create_zero_shot_detector(prompt_generator, detector):
    """
    创建零样本异常检测器实例
    
    Args:
        prompt_generator: 提示词生成器实例
        detector: SAM3异常检测器实例
    
    Returns:
        ZeroShotAnomalyDetector: 零样本异常检测器实例
    """
    return ZeroShotAnomalyDetector(prompt_generator, detector)


def generate_cross_dataset_experiments(source_datasets: List[str] = ['mvtec'], 
                                      target_datasets: List[str] = ['visa']) -> List[Dict]:
    """
    生成跨数据集实验配置
    
    Args:
        source_datasets: 源数据集列表
        target_datasets: 目标数据集列表
    
    Returns:
        list: 实验配置列表
    """
    experiments = []
    
    # 数据集类别映射
    dataset_categories = {
        'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
                 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
                 'transistor', 'wood', 'zipper'],
        'visa': ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
                'pill', 'screw', 'zipper']
    }
    
    # 生成所有可能的源-目标数据集组合
    for source_dataset in source_datasets:
        for target_dataset in target_datasets:
            if source_dataset == target_dataset:
                continue  # 跳过相同数据集的情况
            
            # 获取源数据集和目标数据集的类别
            source_categories = dataset_categories.get(source_dataset, [])
            target_categories = dataset_categories.get(target_dataset, [])
            
            # 生成类别对
            for source_category in source_categories:
                for target_category in target_categories:
                    # 只生成相关的类别对
                    if source_category == target_category:
                        experiments.append({
                            'source_dataset': source_dataset,
                            'target_dataset': target_dataset,
                            'source_category': source_category,
                            'target_category': target_category,
                            'type': 'direct'
                        })
    
    return experiments
