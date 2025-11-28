"""
SAM3异常检测包装器
专门用于基于文本提示的异常检测任务
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from loguru import logger

from .model_builder import build_sam3_image_model
from .model.geometry_encoders import Prompt
from .train.data.sam3_image_dataset import FindQuery, Object, Image as Sam3Image, Datapoint
from .model.box_ops import box_xywh_to_xyxy
from .utils.prompt_engineering import PromptEngine


class SAM3AnomalyPredictor:
    """
    SAM3异常检测预测器
    使用文本提示进行异常检测和分割
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "cuda",
                 checkpoint_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 use_fp16: bool = True):
        """
        初始化SAM3异常检测器
        
        Args:
            model_size: SAM3模型大小 ('base', 'large', 'huge')
            device: 设备 ('cuda', 'cpu')
            checkpoint_path: 预训练权重路径
            confidence_threshold: 置信度阈值
            use_fp16: 是否使用半精度
        """
        self.model_size = model_size
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16
        
        # 初始化提示工程器
        self.prompt_engine = PromptEngine()
        
        # 加载SAM3模型
        self.model = self._load_model(checkpoint_path)
        
        # 设置随机种子确保可复现性
        self.set_seed(122)
        
        logger.info(f"SAM3异常检测器初始化完成，设备: {device}, 模型大小: {model_size}")
    
    def _load_model(self, checkpoint_path: Optional[str] = None) -> nn.Module:
        """加载SAM3模型"""
        try:
            model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                model_size=self.model_size,
                device=self.device,
                eval_mode=True,
                enable_segmentation=True,
                load_from_HF=True if checkpoint_path is None else False
            )
            
            if self.use_fp16 and self.device == "cuda":
                model = model.half()
                
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"加载SAM3模型失败: {e}")
            raise
    
    def set_seed(self, seed: int):
        """设置随机种子确保可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        预处理输入图像
        
        Args:
            image: 输入图像 (路径, PIL Image, 或 numpy array)
            
        Returns:
            预处理后的张量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 调整图像大小到SAM3输入尺寸 (1008x1008)
        target_size = 1008
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 转换为张量并标准化
        image_tensor = np.array(image).astype(np.float32)
        image_tensor = image_tensor / 255.0  # 归一化到 [0, 1]
        
        # SAM3期望的标准化 (mean=0.5, std=0.5)
        image_tensor = (image_tensor - 0.5) / 0.5
        
        # 转换为 CHW 格式
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def create_anomaly_prompts(self, 
                              dataset_name: str = "mvtec",
                              category: str = "bottle",
                              context_image: Optional[np.ndarray] = None) -> List[str]:
        """
        创建异常检测的文本提示
        
        Args:
            dataset_name: 数据集名称
            category: 数据集类别
            context_image: 上下文图像 (可选)
            
        Returns:
            文本提示列表
        """
        return self.prompt_engine.generate_prompts(
            dataset_name=dataset_name,
            category=category,
            context_image=context_image
        )
    
    def predict_anomaly(self, 
                       image: Union[str, Image.Image, np.ndarray],
                       prompts: Optional[List[str]] = None,
                       dataset_name: str = "mvtec",
                       category: str = "bottle") -> Dict[str, Any]:
        """
        对单张图像进行异常检测
        
        Args:
            image: 输入图像
            prompts: 文本提示列表 (可选)
            dataset_name: 数据集名称
            category: 数据集类别
            
        Returns:
            预测结果字典
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image)
        
        # 获取文本提示
        if prompts is None:
            prompts = self.create_anomaly_prompts(dataset_name, category)
        
        # 构建查询
        find_queries = []
        for i, prompt in enumerate(prompts):
            find_query = FindQuery(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=False,
                query_processing_order=i
            )
            find_queries.append(find_query)
        
        # 构建数据点
        objects = [Object(
            bbox=torch.tensor([0.1, 0.1, 0.8, 0.8]),  # 整个图像的边界框
            area=0.64,
            segment=None,
            is_crowd=False
        )]
        
        sam3_image = Sam3Image(
            data=image_tensor,
            objects=objects,
            size=(1008, 1008)
        )
        
        datapoint = Datapoint(
            find_queries=find_queries,
            images=[sam3_image]
        )
        
        # 进行预测
        try:
            with torch.no_grad():
                if self.use_fp16:
                    image_tensor = image_tensor.half()
                
                # 构建几何提示 (使用整个图像作为提示区域)
                geometric_prompt = Prompt(
                    points=None,
                    boxes=None,
                    masks=None
                )
                
                # 调用SAM3模型进行推理
                # 注意：这里需要根据SAM3的实际API调整
                outputs = self._forward_model(image_tensor, find_queries, geometric_prompt)
                
                # 处理输出结果
                results = self._postprocess_outputs(outputs, prompts)
                
                return results
                
        except Exception as e:
            logger.error(f"异常检测推理失败: {e}")
            return self._create_empty_result()
    
    def _forward_model(self, 
                      image_tensor: torch.Tensor, 
                      find_queries: List[FindQuery],
                      geometric_prompt: Prompt) -> Dict[str, Any]:
        """调用SAM3模型进行前向推理"""
        # 这里需要根据SAM3的实际API进行调整
        # 由于SAM3的推理比较复杂，这里提供一个基础框架
        
        # 准备输入数据
        batch = {
            'image': image_tensor,
            'find_queries': find_queries,
            'geometric_prompt': geometric_prompt
        }
        
        # 调用模型
        with torch.no_grad():
            outputs = self.model(batch)
        
        return outputs
    
    def _postprocess_outputs(self, outputs: Dict[str, Any], prompts: List[str]) -> Dict[str, Any]:
        """后处理模型输出"""
        results = {
            'anomaly_scores': [],
            'masks': [],
            'bboxes': [],
            'prompt_matches': prompts,
            'is_anomaly': False,
            'max_anomaly_score': 0.0,
            'best_mask': None,
            'best_bbox': None
        }
        
        try:
            # 提取预测的掩码和分数
            if 'pred_masks' in outputs:
                masks = outputs['pred_masks']
                scores = outputs.get('pred_scores', [0.5] * len(masks))
                
                for mask, score in zip(masks, scores):
                    results['anomaly_scores'].append(score)
                    results['masks'].append(mask.cpu().numpy())
                    
                    # 计算边界框
                    if mask.dim() == 2:
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask.squeeze(0).cpu().numpy()
                    
                    # 找到掩码的边界框
                    coords = np.column_stack(np.where(mask_np > 0.5))
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        results['bboxes'].append(bbox)
                    else:
                        results['bboxes'].append([0, 0, 0, 0])
            
            # 确定是否有异常
            if results['anomaly_scores']:
                max_score = max(results['anomaly_scores'])
                results['max_anomaly_score'] = max_score
                results['is_anomaly'] = max_score > self.confidence_threshold
                
                # 找到最佳掩码
                best_idx = np.argmax(results['anomaly_scores'])
                results['best_mask'] = results['masks'][best_idx]
                results['best_bbox'] = results['bboxes'][best_idx]
        
        except Exception as e:
            logger.warning(f"后处理输出时出错: {e}")
        
        return results
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空的预测结果"""
        return {
            'anomaly_scores': [],
            'masks': [],
            'bboxes': [],
            'prompt_matches': [],
            'is_anomaly': False,
            'max_anomaly_score': 0.0,
            'best_mask': None,
            'best_bbox': None
        }
    
    def batch_predict(self, 
                     images: List[Union[str, Image.Image, np.ndarray]],
                     dataset_name: str = "mvtec",
                     category: str = "bottle",
                     prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量异常检测
        
        Args:
            images: 图像列表
            dataset_name: 数据集名称
            category: 数据集类别
            prompts: 文本提示列表 (可选)
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"处理图像 {i+1}/{len(images)}")
            result = self.predict_anomaly(image, prompts, dataset_name, category)
            results.append(result)
        
        return results
    
    def save_prediction_visualization(self, 
                                    image: Union[str, Image.Image, np.ndarray],
                                    result: Dict[str, Any],
                                    save_path: str,
                                    show_scores: bool = True):
        """
        保存预测结果可视化
        
        Args:
            image: 输入图像
            result: 预测结果
            save_path: 保存路径
            show_scores: 是否显示分数
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        
        # 调整图像大小
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (1008, 1008))
        
        # 创建可视化图像
        vis_image = image_resized.copy()
        
        # 绘制掩码
        if result['best_mask'] is not None:
            mask = result['best_mask']
            if mask.shape != (1008, 1008):
                mask = cv2.resize(mask, (1008, 1008))
            
            # 创建颜色掩码
            colored_mask = np.zeros_like(vis_image)
            colored_mask[:, :, 0] = 255  # 红色
            colored_mask[:, :, 1] = 0
            colored_mask[:, :, 2] = 0
            
            # 应用掩码透明度
            alpha = 0.3
            mask_3d = np.stack([mask] * 3, axis=2)
            vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0, mask_3d)
        
        # 绘制边界框
        if result['best_bbox'] is not None:
            bbox = result['best_bbox']
            x, y, w_bbox, h_bbox = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w_bbox), int(y + h_bbox)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            if show_scores:
                label = f"异常分数: {result['max_anomaly_score']:.3f}"
                cv2.putText(vis_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        logger.info(f"可视化结果已保存到: {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'use_fp16': self.use_fp16,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }