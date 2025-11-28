#!/usr/bin/env python3
"""
快速模块测试脚本
"""

import sys
import traceback

def test_modules():
    results = []
    
    print("=== SamAD3 模块连接测试 ===")
    
    # 测试 1: 模型模块
    try:
        from model.inference import get_sam3_detector
        results.append("✓ model.inference: 成功")
        print("✓ model.inference: 成功")
    except Exception as e:
        results.append(f"✗ model.inference: 失败 - {e}")
        print(f"✗ model.inference: 失败 - {e}")
    
    # 测试 2: 数据集模块
    try:
        from dataset import get_anomaly_dataset
        results.append("✓ dataset: 成功")
        print("✓ dataset: 成功")
    except Exception as e:
        results.append(f"✗ dataset: 失败 - {e}")
        print(f"✗ dataset: 失败 - {e}")
    
    # 测试 3: 提示工程模块
    try:
        from utils.prompt_engineering import get_prompt_generator
        results.append("✓ utils.prompt_engineering: 成功")
        print("✓ utils.prompt_engineering: 成功")
    except Exception as e:
        results.append(f"✗ utils.prompt_engineering: 失败 - {e}")
        print(f"✗ utils.prompt_engineering: 失败 - {e}")
    
    # 测试 4: 评估模块
    try:
        from utils.evaluation import compute_evaluation_metrics
        results.append("✓ utils.evaluation: 成功")
        print("✓ utils.evaluation: 成功")
    except Exception as e:
        results.append(f"✗ utils.evaluation: 失败 - {e}")
        print(f"✗ utils.evaluation: 失败 - {e}")
    
    # 测试 5: 配置模块
    try:
        from utils.config import Config
        results.append("✓ utils.config: 成功")
        print("✓ utils.config: 成功")
    except Exception as e:
        results.append(f"✗ utils.config: 失败 - {e}")
        print(f"✗ utils.config: 失败 - {e}")
    
    # 统计结果
    success_count = len([r for r in results if "✓" in r])
    total_count = len(results)
    
    print(f"\n=== 测试总结 ===")
    print(f"成功: {success_count}/{total_count}")
    
    # 将结果写入文件
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("=== SamAD3 模块连接测试结果 ===\n\n")
        for result in results:
            f.write(result + "\n")
        f.write(f"\n总计: {success_count}/{total_count} 个模块导入成功")
    
    print("测试完成，结果已保存到 test_results.txt")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_modules()
    sys.exit(0 if success else 1)