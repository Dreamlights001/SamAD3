#!/usr/bin/env python3
"""
模块连接测试脚本
检查SamAD3项目中各个模块是否能够正确导入和连接
"""

import sys
import os

def test_imports():
    """测试所有关键模块的导入"""
    results = {}
    
    print("=== SamAD3 模块连接测试 ===\n")
    
    # 测试 1: 模型模块
    try:
        from model.inference import get_sam3_detector
        results['model.inference'] = "✓ 成功"
        print("✓ model.inference 模块导入成功")
        print(f"  - get_sam3_detector: {get_sam3_detector}")
    except Exception as e:
        results['model.inference'] = f"✗ 失败: {e}"
        print(f"✗ model.inference 导入失败: {e}")
    
    # 测试 2: 数据集模块
    try:
        from dataset.base_dataset import get_anomaly_dataset
        results['dataset.base_dataset'] = "✓ 成功"
        print("✓ dataset.base_dataset 模块导入成功")
        print(f"  - get_anomaly_dataset: {get_anomaly_dataset}")
    except Exception as e:
        results['dataset.base_dataset'] = f"✗ 失败: {e}"
        print(f"✗ dataset.base_dataset 导入失败: {e}")
    
    # 测试 3: 提示工程模块
    try:
        from utils.prompt_engineering import get_prompt_generator, AnomalyPromptGenerator
        results['utils.prompt_engineering'] = "✓ 成功"
        print("✓ utils.prompt_engineering 模块导入成功")
        print(f"  - get_prompt_generator: {get_prompt_generator}")
        print(f"  - AnomalyPromptGenerator: {AnomalyPromptGenerator}")
    except Exception as e:
        results['utils.prompt_engineering'] = f"✗ 失败: {e}"
        print(f"✗ utils.prompt_engineering 导入失败: {e}")
    
    # 测试 4: 评估模块
    try:
        from utils.evaluation import compute_evaluation_metrics
        results['utils.evaluation'] = "✓ 成功"
        print("✓ utils.evaluation 模块导入成功")
        print(f"  - compute_evaluation_metrics: {compute_evaluation_metrics}")
    except Exception as e:
        results['utils.evaluation'] = f"✗ 失败: {e}"
        print(f"✗ utils.evaluation 导入失败: {e}")
    
    # 测试 5: 配置模块
    try:
        from utils.config import Config
        results['utils.config'] = "✓ 成功"
        print("✓ utils.config 模块导入成功")
        print(f"  - Config: {Config}")
    except Exception as e:
        results['utils.config'] = f"✗ 失败: {e}"
        print(f"✗ utils.config 导入失败: {e}")
    
    print("\n=== 测试总结 ===")
    success_count = sum(1 for result in results.values() if "✓" in result)
    total_count = len(results)
    print(f"成功: {success_count}/{total_count}")
    
    for module, result in results.items():
        print(f"  {module}: {result}")
    
    return results

def test_class_instantiation():
    """测试关键类的实例化"""
    print("\n=== 类实例化测试 ===\n")
    
    results = {}
    
    # 测试 1: AnomalyPromptGenerator
    try:
        from utils.prompt_engineering import get_prompt_generator
        prompt_gen = get_prompt_generator()
        results['AnomalyPromptGenerator'] = "✓ 成功"
        print("✓ AnomalyPromptGenerator 实例化成功")
        
        # 测试方法
        base_prompts = prompt_gen.get_base_prompts()
        print(f"  - get_base_prompts() 返回: {len(base_prompts)} 个提示词")
        
    except Exception as e:
        results['AnomalyPromptGenerator'] = f"✗ 失败: {e}"
        print(f"✗ AnomalyPromptGenerator 实例化失败: {e}")
    
    # 测试 2: Config
    try:
        from utils.config import Config
        config = Config()
        results['Config'] = "✓ 成功"
        print("✓ Config 实例化成功")
        
    except Exception as e:
        results['Config'] = f"✗ 失败: {e}"
        print(f"✗ Config 实例化失败: {e}")
    
    # 测试 3: SAM3AnomalyDetector (不加载权重)
    try:
        from model.inference import SAM3AnomalyDetector
        results['SAM3AnomalyDetector'] = "✓ 成功"
        print("✓ SAM3AnomalyDetector 类存在")
        
    except Exception as e:
        results['SAM3AnomalyDetector'] = f"✗ 失败: {e}"
        print(f"✗ SAM3AnomalyDetector 导入失败: {e}")
    
    print("\n=== 类实例化总结 ===")
    success_count = sum(1 for result in results.values() if "✓" in result)
    total_count = len(results)
    print(f"成功: {success_count}/{total_count}")
    
    for class_name, result in results.items():
        print(f"  {class_name}: {result}")
    
    return results

def test_file_structure():
    """测试文件结构"""
    print("\n=== 文件结构测试 ===\n")
    
    required_files = [
        "model/inference.py",
        "dataset/base_dataset.py",
        "utils/prompt_engineering.py",
        "utils/evaluation.py",
        "utils/config.py",
        "scripts/test.py",
        "scripts/demo.py",
        "scripts/train.py"
    ]
    
    results = {}
    for file_path in required_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            results[file_path] = "✓ 存在"
            print(f"✓ {file_path} 存在")
        else:
            results[file_path] = "✗ 缺失"
            print(f"✗ {file_path} 缺失")
    
    print("\n=== 文件结构总结 ===")
    success_count = sum(1 for result in results.values() if "✓" in result)
    total_count = len(results)
    print(f"存在: {success_count}/{total_count}")
    
    return results

if __name__ == "__main__":
    print("SamAD3 模块连接检查工具")
    print("=" * 50)
    
    # 运行所有测试
    import_results = test_imports()
    instantiation_results = test_class_instantiation()
    file_results = test_file_structure()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    # 如果有任何失败，退出时返回非零状态码
    all_failures = [
        result for result in import_results.values() 
        if "✗" in result
    ] + [
        result for result in instantiation_results.values() 
        if "✗" in result
    ] + [
        result for result in file_results.values() 
        if "✗" in result
    ]
    
    if all_failures:
        print(f"\n⚠️  发现 {len(all_failures)} 个问题需要修复")
        sys.exit(1)
    else:
        print("\n✅ 所有模块连接正常！")
        sys.exit(0)