"""
SAM3异常检测模型模块

该模块集成了SAM3模型用于异常检测任务，提供了便捷的接口进行零样本异常分割和检测。
"""

from .inference import SAM3AnomalyDetector, get_sam3_detector

__all__ = [
    "SAM3AnomalyDetector",
    "get_sam3_detector"
]

__version__ = "0.1.0"
