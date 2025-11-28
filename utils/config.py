import os
import yaml
import argparse
from typing import Dict, Any, Optional, List
import logging
import copy

# 设置日志
logger = logging.getLogger(__name__)

class Config:
    """
    配置管理类，用于加载、解析和管理YAML配置文件
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置类
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # 如果提供了配置文件路径，加载配置
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
        
        Raises:
            FileNotFoundError: 如果配置文件不存在
            yaml.YAMLError: 如果YAML解析失败
        """
        config_path = os.path.expanduser(config_path)  # 扩展~路径
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"成功加载配置文件: {config_path}")
            # 扩展所有路径中的~符号
            self._expand_paths()
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _expand_paths(self) -> None:
        """
        递归扩展配置中的所有路径
        """
        def _expand(obj: Any) -> Any:
            if isinstance(obj, str):
                return os.path.expanduser(obj)
            elif isinstance(obj, dict):
                return {key: _expand(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [_expand(item) for item in obj]
            else:
                return obj
        
        self.config = _expand(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键路径访问
        
        Args:
            key: 配置键，支持点号分隔的嵌套路径，如 'model.sam3.device'
            default: 默认值，如果键不存在
        
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if not isinstance(value, dict):
                    raise KeyError(f"键 '{k}' 在路径 '{key}' 中不存在")
                value = value[k]
            return value
        except KeyError:
            logger.warning(f"配置键 '{key}' 不存在，返回默认值")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持嵌套键路径
        
        Args:
            key: 配置键，支持点号分隔的嵌套路径
            value: 新的值
        """
        keys = key.split('.')
        config = self.config
        
        # 遍历除最后一个键之外的所有键
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                raise ValueError(f"键 '{k}' 已经存在但不是字典类型")
            config = config[k]
        
        # 设置最后一个键的值
        config[keys[-1]] = value
        logger.debug(f"设置配置键 '{key}' = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        更新配置字典
        
        Args:
            updates: 要更新的配置字典
        """
        def _update_recursive(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in updates.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key] = _update_recursive(config[key], value)
                else:
                    config[key] = value
            return config
        
        self.config = _update_recursive(self.config, updates)
        logger.debug(f"更新配置: {updates}")
    
    def from_args(self, args: argparse.Namespace) -> None:
        """
        从命令行参数更新配置
        
        Args:
            args: 命令行参数命名空间
        """
        # 转换命名空间为字典
        args_dict = vars(args)
        
        # 过滤掉None值
        args_dict = {k: v for k, v in args_dict.items() if v is not None}
        
        # 更新配置
        self.update(args_dict)
        logger.info(f"从命令行参数更新配置")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典的深拷贝
        
        Returns:
            配置字典的深拷贝
        """
        return copy.deepcopy(self.config)
    
    def save(self, save_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        save_path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            bool: 配置是否有效
        """
        # 基础验证
        required_sections = ['paths', 'model', 'prompt']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"缺少必需的配置部分: {section}")
                return False
        
        # 验证路径配置
        if 'paths' in self.config:
            paths = self.config['paths']
            # 验证预训练权重路径（如果指定了）
            if 'pretrained_weights' in paths and paths['pretrained_weights']:
                pretrained_path = os.path.expanduser(paths['pretrained_weights'])
                if not os.path.exists(pretrained_path):
                    logger.warning(f"预训练权重文件不存在: {pretrained_path}")
            
            # 验证数据集根目录
            if 'datasets_root' in paths:
                datasets_root = os.path.expanduser(paths['datasets_root'])
                if not os.path.exists(datasets_root):
                    logger.warning(f"数据集根目录不存在: {datasets_root}")
        
        # 验证模型配置
        if 'model' in self.config and 'sam3' in self.config['model']:
            sam3_config = self.config['model']['sam3']
            if 'device' not in sam3_config:
                logger.error("缺少SAM3设备配置")
                return False
        
        logger.info("配置验证通过")
        return True
    
    def get_paths(self) -> Dict[str, str]:
        """
        获取所有路径配置
        
        Returns:
            路径配置字典
        """
        return self.get('paths', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.get('model', {})
    
    def get_prompts(self, dataset: Optional[str] = None, 
                   category: Optional[str] = None) -> List[str]:
        """
        获取提示词列表
        
        Args:
            dataset: 数据集名称
            category: 类别名称
        
        Returns:
            提示词列表
        """
        # 获取基础提示词
        base_prompts = self.get('prompt.base_prompts', [])
        
        # 如果指定了数据集和类别，获取特定的提示词
        if dataset and category:
            dataset_prompts = self.get(f'prompt.dataset_specific.{dataset}.{category}', [])
            return base_prompts + dataset_prompts
        
        return base_prompts
    
    def __str__(self) -> str:
        """
        返回配置的字符串表示
        """
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)


def get_default_config() -> Config:
    """
    获取默认配置
    
    Returns:
        默认配置对象
    """
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config_path = os.path.join(project_root, 'configs', 'base_config.yaml')
    
    # 如果存在默认配置文件，加载它
    if os.path.exists(default_config_path):
        return Config(default_config_path)
    
    # 否则返回空配置
    logger.warning(f"未找到默认配置文件: {default_config_path}")
    return Config()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        命令行参数命名空间
    """
    parser = argparse.ArgumentParser(description='SAM3异常检测')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default='./configs/base_config.yaml',
                      help='配置文件路径')
    
    # 路径参数
    parser.add_argument('--pretrained-weights', type=str,
                      help='预训练权重路径')
    parser.add_argument('--datasets-root', type=str,
                      help='数据集根目录')
    parser.add_argument('--output-dir', type=str,
                      help='输出目录')
    
    # 模型参数
    parser.add_argument('--model-size', type=str, choices=['base', 'large', 'huge'],
                      help='模型大小')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      help='设备')
    parser.add_argument('--use-fp16', action='store_true',
                      help='使用半精度')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'],
                      help='数据集名称')
    parser.add_argument('--category', type=str,
                      help='数据集类别')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test',
                      help='数据集分割')
    
    # 推理参数
    parser.add_argument('--image-path', type=str,
                      help='推理图像路径')
    parser.add_argument('--threshold', type=float,
                      help='异常检测阈值')
    
    # 零样本参数
    parser.add_argument('--zero-shot', action='store_true',
                      help='启用零样本模式')
    parser.add_argument('--source-dataset', type=str, choices=['mvtec', 'visa'],
                      help='源数据集')
    parser.add_argument('--target-dataset', type=str, choices=['mvtec', 'visa'],
                      help='目标数据集')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int,
                      help='批量大小')
    parser.add_argument('--learning-rate', type=float,
                      help='学习率')
    parser.add_argument('--num-epochs', type=int,
                      help='训练轮数')
    parser.add_argument('--fine-tune', action='store_true',
                      help='启用微调模式')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                      help='启用可视化')
    parser.add_argument('--save-visualizations', action='store_true',
                      help='保存可视化结果')
    parser.add_argument('--tsne', action='store_true',
                      help='启用t-SNE可视化')
    
    # 其他参数
    parser.add_argument('--seed', type=int,
                      help='随机种子')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='日志级别')
    
    return parser.parse_args()


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # 移除所有现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 如果提供了日志文件，添加文件处理器
    if log_file:
        log_file = os.path.expanduser(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"日志配置完成，级别: {log_level}")


def set_random_seed(seed: int = 122, deterministic: bool = True) -> None:
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法
    """
    import random
    import numpy as np
    import torch
    
    # 设置Python随机种子
    random.seed(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置确定性
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"随机种子设置为: {seed}, 确定性: {deterministic}")


def create_directory_structure(config: Config) -> None:
    """
    根据配置创建目录结构
    
    Args:
        config: 配置对象
    """
    paths = config.get_paths()
    
    # 创建所有必要的目录
    directories = [
        paths.get('output_dir', './output'),
        paths.get('log_dir', './logs'),
        paths.get('visualization_dir', './assets/visualizations'),
        # 确保configs目录存在
        './configs'
    ]
    
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
