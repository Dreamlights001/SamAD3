from typing import List, Dict, Optional
import json
import os

class AnomalyPromptGenerator:
    """
    异常检测提示词生成器，用于生成针对不同数据集和场景的提示词
    支持基础提示、混合提示和零样本提示生成
    """
    
    # 默认的基础异常提示词
    DEFAULT_ANOMALY_PROMPTS = [
        "anomaly", "defect", "damage", "imperfection", 
        "fault", "flaw", "error", "problem",
        "irregularity", "abnormality", "distortion", "contamination"
    ]
    
    # 数据集特定的提示词映射
    DATASET_SPECIFIC_PROMPTS = {
        "mvtec": {
            "bottle": ["broken glass", "liquid leakage", "contamination", "chip"],
            "cable": ["bent wire", "insulation damage", "cut wire"],
            "capsule": ["crack", "pill missing", "discoloration", "deformation"],
            "carpet": ["stain", "hole", "wrinkle", "tear"],
            "grid": ["broken grid", "missing line", "disconnection"],
            "hazelnut": ["crack", "broken shell", "hole"],
            "leather": ["scratch", "cut", "discoloration", "hole"],
            "metal_nut": ["thread damage", "broken nut", "scratch"],
            "pill": ["broken pill", "crack", "missing pill"],
            "screw": ["thread damage", "missing head", "bent"],
            "tile": ["crack", "scratch", "chip", "stain"],
            "toothbrush": ["bent bristles", "missing bristles"],
            "transistor": ["broken lead", "missing component"],
            "wood": ["crack", "hole", "stain", "scratch"],
            "zipper": ["broken slider", "misaligned teeth", "missing pull"]
        },
        "visa": {
            "candle": ["melted candle", "broken wick", "deformation"],
            "capsules": ["broken capsule", "missing capsule", "discoloration"],
            "cashew": ["broken cashew", "shell fragment", "discoloration"],
            "chewinggum": ["deformed gum", "stain", "damage"],
            "fryum": ["broken fryum", "irregular shape", "discoloration"],
            "macaroni1": ["broken pasta", "irregular shape", "stain"],
            "macaroni2": ["broken pasta", "irregular shape", "stain"],
            "pcb1": ["missing component", "soldering defect", "broken trace"],
            "pcb2": ["missing component", "soldering defect", "broken trace"],
            "pcb3": ["missing component", "soldering defect", "broken trace"],
            "pcb4": ["missing component", "soldering defect", "broken trace"],
            "pipe_fryum": ["broken fryum", "irregular shape", "discoloration"]
        }
    }
    
    def __init__(self, custom_prompts_file: Optional[str] = None):
        """
        初始化提示词生成器
        
        Args:
            custom_prompts_file: 自定义提示词文件路径（JSON格式）
        """
        self.base_prompts = self.DEFAULT_ANOMALY_PROMPTS.copy()
        self.dataset_specific = self.DATASET_SPECIFIC_PROMPTS.copy()
        
        # 加载自定义提示词
        if custom_prompts_file and os.path.exists(custom_prompts_file):
            self._load_custom_prompts(custom_prompts_file)
    
    def _load_custom_prompts(self, file_path: str):
        """
        加载自定义提示词
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
                
            if 'base_prompts' in custom_data:
                self.base_prompts.extend(custom_data['base_prompts'])
                self.base_prompts = list(set(self.base_prompts))  # 去重
            
            if 'dataset_specific' in custom_data:
                for dataset, categories in custom_data['dataset_specific'].items():
                    if dataset not in self.dataset_specific:
                        self.dataset_specific[dataset] = {}
                    self.dataset_specific[dataset].update(categories)
        except Exception as e:
            print(f"警告: 加载自定义提示词文件失败: {e}")
    
    def get_base_prompts(self) -> List[str]:
        """
        获取基础异常提示词
        """
        return self.base_prompts
    
    def get_dataset_prompts(self, dataset_name: str, category: Optional[str] = None) -> List[str]:
        """
        获取特定数据集和类别的提示词
        
        Args:
            dataset_name: 数据集名称（如'mvtec', 'visa'）
            category: 数据集类别（如'bottle', 'candle'）
            
        Returns:
            提示词列表
        """
        prompts = self.base_prompts.copy()
        
        # 添加数据集特定提示词
        if dataset_name in self.dataset_specific:
            if category and category in self.dataset_specific[dataset_name]:
                prompts.extend(self.dataset_specific[dataset_name][category])
        
        return list(set(prompts))  # 去重
    
    def generate_mixed_prompts(self,
                              base_prompts: List[str],
                              context_info: Optional[str] = None,
                              dataset_name: Optional[str] = None,
                              category: Optional[str] = None) -> List[str]:
        """
        生成混合提示词，结合基础提示、数据集特定提示和上下文信息
        
        Args:
            base_prompts: 基础提示词列表
            context_info: 上下文信息（如"在金属表面"）
            dataset_name: 数据集名称
            category: 数据集类别
            
        Returns:
            混合后的提示词列表
        """
        # 获取完整提示词列表
        all_prompts = base_prompts.copy()
        
        # 添加数据集特定提示词
        if dataset_name:
            dataset_prompts = self.get_dataset_prompts(dataset_name, category)
            all_prompts.extend(dataset_prompts)
        
        # 去重
        all_prompts = list(set(all_prompts))
        
        # 添加上下文信息
        mixed_prompts = all_prompts.copy()
        if context_info:
            for prompt in all_prompts:
                mixed_prompts.append(f"{context_info} {prompt}")
        
        return mixed_prompts
    
    def generate_zeroshot_prompts(self,
                                source_dataset: str,
                                target_dataset: str,
                                source_category: Optional[str] = None,
                                target_category: Optional[str] = None,
                                context_info: Optional[str] = None) -> List[str]:
        """
        生成零样本场景的提示词，结合源数据集和目标数据集的信息
        
        Args:
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            source_category: 源数据集类别
            target_category: 目标数据集类别
            context_info: 额外的上下文信息
            
        Returns:
            零样本提示词列表
        """
        # 获取源数据集和目标数据集的提示词
        source_prompts = self.get_dataset_prompts(source_dataset, source_category)
        target_prompts = self.get_dataset_prompts(target_dataset, target_category)
        
        # 合并并去重
        combined_prompts = list(set(source_prompts + target_prompts))
        
        # 添加混合上下文提示
        if context_info:
            return self.generate_mixed_prompts(combined_prompts, context_info)
        
        return combined_prompts

# 创建一个简单的工具函数，用于获取提示词生成器实例
def get_prompt_generator(custom_prompts_file: Optional[str] = None) -> AnomalyPromptGenerator:
    """
    获取异常提示词生成器实例
    
    Args:
        custom_prompts_file: 自定义提示词文件路径
        
    Returns:
        提示词生成器实例
    """
    return AnomalyPromptGenerator(custom_prompts_file)
