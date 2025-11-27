# 项目导入测试脚本
# 此脚本用于验证项目的导入关系和基本结构是否正确

try:
    print("开始测试项目导入...")
    
    # 测试配置文件导入
    print("测试配置文件导入...")
    import config
    print(f"配置文件导入成功，数据集路径: {config.DATASET_ROOT}")
    
    # 测试模型模块导入
    print("测试模型模块导入...")
    from models.sam3_model import SAM3Model
    print("模型模块导入成功")
    
    # 测试数据加载模块导入
    print("测试数据加载模块导入...")
    from data_loaders.dataset_loader import DatasetLoader, MVTecDatasetLoader, VisADatasetLoader
    from data_loaders.data_pipeline import DataPipeline, create_data_pipeline
    print("数据加载模块导入成功")
    
    # 测试工具函数导入
    print("测试工具函数导入...")
    from utils.metrics import calculate_iou, evaluate_prediction
    from utils.image_processing import preprocess_image
    from utils.data_augmentation import augment_image
    print("工具函数导入成功")
    
    # 测试主程序导入
    print("测试主程序导入...")
    from defect_detector import DefectDetector
    print("主程序导入成功")
    
    print("\n✅ 所有模块导入测试通过！")
    print("注意：完整功能测试需要SAM3模型访问权限和实际数据集。")
    print("可以运行 'python test_imports.py' 验证基本结构是否正确。")
    
except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print("请检查安装是否正确，尤其是依赖包和项目结构。")
except Exception as e:
    print(f"\n❌ 发生错误: {e}")
