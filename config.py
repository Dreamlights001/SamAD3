# 项目配置文件
import os
import torch

# 数据集路径配置
DATASET_ROOT = os.path.expanduser("~/autodl-tmp/datasets")

# 预训练权重路径
MODEL_WEIGHTS_PATH = os.path.expanduser("~/autodl-tmp/pre-training")

# 支持的数据集
supported_datasets = ["mvtec", "visa", "btad", "DAGM_KaggleUpload"]

# 模型配置
MODEL_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
    "max_detections": 100,
}

# 日志配置
LOG_LEVEL = "INFO"

# 输出结果保存路径
OUTPUT_DIR = "./outputs"

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
