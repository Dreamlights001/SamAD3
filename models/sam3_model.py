# SAM3模型加载和推理模块
import torch
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import MODEL_WEIGHTS_PATH, MODEL_CONFIG
from utils.image_processing import preprocess_image, postprocess_mask

class SAM3Model:
    """SAM3模型封装类"""
    def __init__(self):
        """初始化模型"""
        self.device = MODEL_CONFIG["device"]
        self.confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        self.iou_threshold = MODEL_CONFIG["iou_threshold"]
        self.max_detections = MODEL_CONFIG["max_detections"]
        
        # 加载模型和处理器
        print(f"加载SAM3模型到{self.device}...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成")
    
    def set_image(self, image):
        """设置输入图像"""
        return self.processor.set_image(image)
    
    def predict_with_text(self, state, text_prompt):
        """使用文本提示进行预测"""
        with torch.no_grad():
            output = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        
        # 过滤低置信度的预测
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # 保留置信度大于阈值的预测
        valid_indices = scores > self.confidence_threshold
        masks = masks[valid_indices]
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        
        # 限制最大检测数量
        if len(scores) > self.max_detections:
            top_indices = torch.argsort(scores, descending=True)[:self.max_detections]
            masks = masks[top_indices]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
        
        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores
        }
    
    def predict_defect(self, image, defect_prompt="defect flaw scratch damage", threshold=0.5):
        """预测图像中的缺陷"""
        original_size = image.size
        
        # 预处理图像
        processed_image = preprocess_image(image)
        
        # 设置图像
        state = self.set_image(processed_image)
        
        # 使用缺陷相关的文本提示进行预测
        results = self.predict_with_text(state, defect_prompt)
        
        # 合并所有掩码
        if len(results["masks"]) > 0:
            # 选择置信度最高的掩码
            best_mask_idx = torch.argmax(results["scores"])
            pred_mask = results["masks"][best_mask_idx].cpu().numpy()
            
            # 后处理掩码，恢复到原始大小
            final_mask = postprocess_mask(pred_mask, original_size)
            
            # 根据阈值调整掩码
            final_mask = final_mask > threshold
        else:
            # 如果没有检测到缺陷，返回空掩码
            final_mask = np.zeros(original_size[::-1], dtype=bool)
        
        return {
            "mask": final_mask,
            "scores": results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([]),
            "boxes": results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
        }
    
    def batch_predict(self, images, defect_prompt="defect flaw scratch damage"):
        """批量预测图像中的缺陷"""
        results = []
        for image in images:
            result = self.predict_defect(image, defect_prompt)
            results.append(result)
        return results
